clc; clear; close all;
rng(42);
%% SINGLE BOUNDARY LAYER CASE

eta=0.1; n_unif=1500;
nu=0.01; w_f = 0.4972; w_mu = 0.97218; w_tau = 0.39696; w_lam =  0.50471; 
% nu=0.001; w_f = 0.50495; w_mu = 0.98715; w_tau = 0.35389; w_lam =  0.75617;  
% nu=0.0001; w_f = 0.50432; w_mu = 0.99898; w_tau = 0.024759; w_lam =  0.70484;  
J_val = KAPI_net(nu, eta, n_unif, w_f, w_mu, w_tau, w_lam);


%% Helper functions
function J_val = KAPI_net(nu, eta, n_unif, w_f, w_mu, w_tau, w_lam)
    % Create parameter structures
    fixed_params = struct('nu', nu, 'eta', eta, 'n_unif', n_unif);
    tunable_params = struct('w_f', w_f, 'w_mu', w_mu, 'w_tau', w_tau, 'w_lam', w_lam);
    
    % Call the function to generate the output cell
    Data_and_Bases = Data_and_Bases_1D(fixed_params, tunable_params);
    
    X_pde = Data_and_Bases{1};
    alpha_star = Data_and_Bases{2};
    sig = Data_and_Bases{3};
    
    save('alpha_and_sig_kapi.mat', 'alpha_star', 'sig');

    X_bc = [0; 1];
    
    % PIELM input layer parameters
    m = 1./(sqrt(2) * sig);
    alpha = -m .* alpha_star;
    
    % Design Matrix
    [H, b] = CONSTRUCT_H_AND_b(X_pde, X_bc, m, alpha, nu);
    
    % Least square solution
    c = pinv(H) * b;  % Least-squares solution
    
    X_val = create_validation_set(X_pde, 0.2, true);

    plot_dataset_distributions(X_pde, X_val, 'FIG_VALIDATION_DISTRIBUTION.png');

    % Compute the loss (L-infty norm of residuals)
    [H, b] = CONSTRUCT_H_AND_b(X_val, X_bc, m, alpha, nu);
    J_val = norm(abs(H * c - b), Inf); 
    J_val = log10(J_val);
    fprintf('Validation error (log10 scale): J_val = %.4f\n', J_val);

    %% RIGOROUS OVERFIT CHECK
    X_pde_test=[linspace(0, 0.9, 4*n_unif)';linspace(0.9, 1, n_unif)'];
    X_pde_test=sort(X_pde_test);

    u_pielm = PIELM_SOLUTION(X_pde_test, m, alpha, c);
    u_exact = EXACT_SOLUTION(X_pde_test, nu);
    plot_PIELM_vs_Exact(X_pde_test, u_exact, u_pielm, nu)
end

function output_cell = Data_and_Bases_1D(fixed_params, tunable_params)
    % Extract fixed parameters
    nu = fixed_params.nu;
    eta = fixed_params.eta;
    n_unif = fixed_params.n_unif;
    

    rng(42);

    % Extract tunable parameters
    w_f = tunable_params.w_f;
    w_mu = tunable_params.w_mu;
    w_tau = tunable_params.w_tau;
    w_lam = tunable_params.w_lam;

    %% Global Part (Fixed)
    % (A) X_pde (Uniform component)
    x_unif = linspace(0, 1, n_unif)';

    % (B) RBF-Center (Uniform component)
    alpha_star_unif = linspace(0, 1, round(0.5*n_unif))';

    % (C) RBF-Width (Uniform component)
    sig_max = 10*(2/n_unif)
    sig_unif = sig_max*ones(size(alpha_star_unif));

    %% Local Part (Adaptive)
    eta_tau = eta * w_tau/3;
    n_adap = round(0.35*n_unif .* w_f ./ sum(w_f)); % MAXIMUM 35%!

    inv_sig_ref = 1/(sqrt(2)*sig_max);

    x_gauss = [];
    alpha_star_gauss = [];
    sig_gauss = [];

    for i = 1:length(w_mu)
        n_i = n_adap(i);  
        lam_i = w_lam(i);

        raw_samples = w_mu(i) + eta_tau(i) * randn(n_i, 1);
        inv_sig_nu_i = inv_sig_ref * exp((1+lam_i) * -log10(nu));
        inv_sig_range_i = unifrnd(-inv_sig_nu_i/2, inv_sig_nu_i/2, 1, n_i);

        reflected_samples = abs(mod(raw_samples, 2));          
        reflected_samples = min(reflected_samples, 2 - reflected_samples); 

        raw_bandwidths = 1./abs(sqrt(2)*inv_sig_range_i)';
        reflected = sig_max * abs(mod(raw_bandwidths/sig_max, 2));
        folded = min(reflected, 2*sig_max - reflected);
        
        x_gauss = [x_gauss; reflected_samples];
        alpha_star_gauss = [alpha_star_gauss; raw_samples];
        sig_gauss = [sig_gauss; folded];
    end

    %% Combine Uniform and Adaptive Points
    X_pde = [x_unif; x_gauss];
    alpha_star = [alpha_star_unif; alpha_star_gauss];
    sig = [sig_unif; sig_gauss];

    scatter_centers_vs_widths(alpha_star, sig, 'F2_RBF_1D_centers.png');
    
    %% Create output cell
    output_cell = {X_pde, alpha_star, sig};
end

function [H, b] = CONSTRUCT_H_AND_b(X_pde, X_bc, m, alpha, nu)
    Nc = length(X_pde);
    NN = length(alpha);
    
    % Initialize LHS_PDE (PDE residuals)
    LHS_PDE = zeros(Nc, NN);
    for k = 1:Nc
        X_k = X_pde(k, :);
        Phi = exp(-(X_k * m' + alpha').^2);  % Vectorized computation

        LHS_PDE(k, :) = -2 * Phi .* (m' .* (X_k * m' + alpha')) ...
                        + 2 * nu * Phi .* (m'.^2) .* (1 - 2 * (X_k * m' + alpha').^2);

    end

    % Initialize LHS_BC (BC residuals)
    LHS_BC = zeros(2, NN);
    for k = 1:2
        X_k = X_bc(k, :);
        Phi = exp(-(X_k * m' + alpha').^2);  % Vectorized computation
        LHS_BC(k, :) = Phi;
    end

    % Combine into H
    H = [LHS_PDE; LHS_BC];

    % Construct b: [zeros(Nc, 1); BC values]
    b = [zeros(Nc, 1); 0; 1];  % PDE RHS = 0, BCs: u(0) = 0, u(1) = 1
end

function X_val = create_validation_set(X_pde, validation_ratio, ensure_unique)
rng(42);
%CREATE_VALIDATION_SET Creates validation set preserving X_pde distribution
%   X_val = create_validation_set(X_pde, validation_ratio, ensure_unique)
%
% Inputs:
%   - X_pde: Column vector (Nx1) of coordinates in [0,1]
%   - validation_ratio: Fraction for validation set (default: 0.2)
%   - ensure_unique: Remove duplicates (default: true)
%
% Output:
%   - X_val: Validation set with same distribution as X_pde

% Set defaults
if nargin < 2
    validation_ratio = 0.2; % 20% validation by default
end
if nargin < 3
    ensure_unique = true; % Ensure unique points by default
end

% Verify ratio is valid
if validation_ratio >= 1
    error('Validation ratio must be < 1');
end

% Step 1: Estimate probability density (Kernel Density Estimation)
[pdf_values, ~] = ksdensity(X_pde, X_pde, 'Function', 'pdf');

% Step 2: Compute sampling weights (higher weight for sparser regions)
weights = 1 ./ (pdf_values + eps); % eps avoids division by zero
weights = weights / sum(weights);  % Normalize to probabilities

% Create indices for all data points
all_indices = 1:length(X_pde);

% Step 3: Sample validation set
n_val = round(validation_ratio * length(X_pde));
val_indices = randsample(all_indices, n_val, true, weights);
X_val = X_pde(val_indices);

% Step 4: Ensure uniqueness (optional)
if ensure_unique
    X_val = unique(X_val);
end
end


function plot_dataset_distributions(X_pde, X_val, save_path)
    % Create figure with white background
    fig = figure('Units', 'inches', 'Position', [0 0 8 6], 'Color', 'w');
    set(fig, 'DefaultAxesFontSize', 14);

    % Color scheme
    colors = struct(...
        'pde', [0.2 0.4 0.8], ...  % Blue for original data
        'val', [0.8 0.2 0.2] ...   % Red for validation
    );

    % Define consistent bin edges based on X_pde
    binEdges = linspace(min(X_pde), max(X_pde), 51); % 50 bins

    % Subplot 1: Original X_pde
    subplot(2, 1, 1);
    h1 = histogram(X_pde, binEdges, 'Normalization', 'pdf', ...
        'FaceColor', colors.pde, 'EdgeColor', 'none');
    title('Training $\mathbf{X}_{\mathrm{pde}}$ Density', ...
        'Interpreter', 'LaTeX', 'FontSize', 16);
    ylabel('PDF', 'Interpreter', 'LaTeX');
    grid on; box on;

    % Subplot 2: Validation set X_val
    subplot(2, 1, 2);
    h2 = histogram(X_val, binEdges, 'Normalization', 'pdf', ...
        'FaceColor', colors.val, 'EdgeColor', 'none');
    title('Validation Set $\mathbf{X}_{\mathrm{val}}$ Density', ...
        'Interpreter', 'LaTeX', 'FontSize', 16);
    xlabel('$\mu$', 'Interpreter', 'LaTeX');
    ylabel('PDF', 'Interpreter', 'LaTeX');
    grid on; box on;

    % Adjust subplot spacing
    set(gcf, 'Units', 'inches', 'Position', [0 0 8 6]);

    % Save if path provided
    if nargin > 2 && ~isempty(save_path)
        print(save_path, '-dpng', '-r300');
    end
end



%% PIELM solution 
function u_pielm = PIELM_SOLUTION(X, m, alpha, c)
    % Compute PIELM solution at points X using Gaussian basis and weights c
    u_pielm = zeros(size(X));
    for k = 1:length(X)
        z_sqr = (m' * X(k) + alpha').^2;
        u_pielm(k) = exp(-z_sqr) * c;
    end
end



%% Exact solution
function u_exact = EXACT_SOLUTION(X, nu)
    % Numerically stable exact solution for 0 ≤ X ≤ 1
    % Handles cases from nu = 1e-15 to nu = 1
    
    % Threshold where exp(1/nu) would overflow (≈1/709)
    overflow_threshold = 1 / log(realmax);
    
    if nu > overflow_threshold
        % Standard calculation for moderate nu
        u_exact = expm1(X ./ nu) ./ expm1(1 / nu);
    else
        % Reformulated calculation for very small nu
        exponent = (X - 1) ./ nu;
        
        % Set values below machine epsilon to 0
        threshold = -log(eps(class(X))); % ≈36 for double
        u_exact = exp(exponent);
        u_exact(exponent < -threshold) = 0;
        
        % Force boundary condition at X=1
        u_exact(X == 1) = 1;  % Exact match for grid points
    end
    
    % Handle floating-point X≈1 cases (optional)
    if any(abs(X(:)-1) < eps(class(X)) & X(:) ~= 1)
        u_exact(abs(X-1) < eps(class(X))) = 1;
    end
end



function plot_PIELM_vs_Exact(X_test, u_exact, u_kapi, nu, filename)
% PLOT_PIELM_VS_EXACT  Compare exact and predicted solutions with error on a single plot
%
%   INPUTS:
%     X_test   - column vector of test locations
%     u_exact  - exact solution evaluated at X_test
%     u_kapi   - predicted solution from PIELM/KAPI at X_test
%     nu       - viscosity parameter value
%     filename - name of PNG file to save (without extension)
%
%   OUTPUT:
%     Saves a 300 DPI PNG plot with solution and error overlay

    % Ensure inputs are column vectors
    X_test = X_test(:);
    u_exact = u_exact(:);
    u_kapi = u_kapi(:); 

    % Compute absolute error
    abs_error = abs(u_exact - u_kapi);

    % Create figure
    figure('Units', 'inches', 'Position', [1, 1, 8, 4]);

    % Plot true and predicted solutions (left y-axis)
    yyaxis left
    plot(X_test, u_exact, 'b-', 'LineWidth', 2); hold on;
    plot(X_test, u_kapi, 'r--', 'LineWidth', 2); xlim([0,1.05]);
    ylabel_str = ['$u(x;\, \nu=' sprintf('%.2f', nu) ')$'];%%%%%%%%%%%%%%%%%%%%%%%
    ylabel(ylabel_str, 'Interpreter', 'latex', 'FontSize', 20);
    ylim([min([u_exact; u_kapi])-0.1, max([u_exact; u_kapi]) + 0.1]);
    ax = gca;
    ax.YColor = 'k';  % Make y-axis black for clarity

    % Plot absolute error (right y-axis)
    yyaxis right
    area(X_test, abs_error, 'FaceColor', [0.1 0.6 0.1], ...
         'FaceAlpha', 0.3, 'EdgeColor', 'none');  % Transparent green fill
    ylabel('$|u_{\mathrm{exact}} - u_{\mathrm{pred}}|$', 'Interpreter', 'latex', 'FontSize', 20);
    ax = gca;
    ax.YColor = [0.1 0.6 0.1];

    % Common x-label
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20);

    % Legend (manual to distinguish axes)
    legend({'True', 'Predicted', 'Abs. Error'}, 'Location', 'northwest', ...
           'Interpreter', 'latex', 'FontSize', 18);
    ylim([min(abs_error), max(abs_error) ]);
    
    % Grid and font
    grid on;
    set(gca, 'FontSize', 20);

    % Save the figure
    if nargin < 5
        filename = 'F1_PIELM_vs_Exact_Overlay';
    end
    print(gcf, filename, '-dpng', '-r300');
end



function scatter_centers_vs_widths(alpha_star, sig, save_path)
    % Match dimensions of the 1st column figure (12x4 inches)
    figure('Units', 'inches', 'Position', [1, 1, 8, 4], ...
           'Color', 'w', 'Name', 'RBF Centers vs Widths');

    % Scatter plot with adjusted marker size
    scatter(alpha_star, sig, 10, 'ko', 'filled'); % Reduced marker size for clarity
    xlim([0,1.05]);
    grid on;

    % Labels and title (match 1st column's font size)
    xlabel('$\alpha^*$ ', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 20, 'LineWidth', 1.2, 'Box', 'off'); % Match subplot styling

    % Save with identical resolution (300 DPI)
    if nargin > 2 && ~isempty(save_path)
        print(gcf, save_path, '-dpng', '-r300');
        fprintf('✅ Saved resized scatter plot to: %s\n', save_path);
    end
end

