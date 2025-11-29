clc; clear; close all;
rng(42);

%% Bayesian Optimization Setup for KAPI_net
% Fixed parameters
nu = 0.001; % Specify nu here

eta = 0.1;n_unif = 1500;
% Define the objective function for bayesopt
objectiveFcn = @(w) KAPI_net_wrapper(nu, eta, n_unif, w);

% Define optimization variables (works till nu = [1e-2, 1e-3])
optimVars = [
    optimizableVariable('w1', [0.495, 0.505]), ...
    optimizableVariable('w2', [0.9, 0.99]), ...
    optimizableVariable('w3', [0.1, 0.4]), ...
    optimizableVariable('w4', [0.5, 1])
];

% % Define optimization variables (tighter bounds for (w2, w3) for very low nu = 1e-4)
% optimVars = [
%     optimizableVariable('w1', [0.495, 0.505]), ...
%     optimizableVariable('w2', [0.99, 0.999]), ...
%     optimizableVariable('w3', [0.01, 0.1]), ...
%     optimizableVariable('w4', [0.5, 1])
% ];

% Run Bayesian Optimization
tic;
results = bayesopt(...
    objectiveFcn, ...
    optimVars, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 200, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', false, ...
    'PlotFcn', [], ...
    'Verbose', 0, ...
    'OutputFcn', @stopIfNoImprovement);
toc;

%% Display and save best results
bestWeights = table2array(results.XAtMinObjective);
bestLoss = results.MinObjective;

disp('Optimization complete!');
disp(['Optimal weights: ', num2str(bestWeights)]);
disp(['Minimum loss J: ', num2str(bestLoss)]);

save('optimization_results.mat', 'nu', 'bestWeights','bestLoss');

% Plot Training History
KAPI_Training_History(results);


%% Helper function to convert bayesopt format to KAPI_net inputs
function J = KAPI_net_wrapper(nu, eta, n_unif, w)
    w_f = w.w1;
    w_mu = w.w2;
    w_tau = w.w3;
    w_lam = w.w4;
    
    % Call your original KAPI_net function here
    J = KAPI_net(nu, eta, n_unif, w_f, w_mu, w_tau, w_lam);
end


%% Your original KAPI_net function (unchanged)
function J = KAPI_net(nu, eta, n_unif, w_f, w_mu, w_tau, w_lam)
    % Create parameter structures
    fixed_params = struct('nu', nu, 'eta', eta, 'n_unif', n_unif);
    tunable_params = struct('w_f', w_f, 'w_mu', w_mu, 'w_tau', w_tau, 'w_lam', w_lam);
    
    % Call the function to generate the output cell
    Data_and_Bases = Data_and_Bases_1D(fixed_params, tunable_params);
    
    X_pde = Data_and_Bases{1};
    alpha_star = Data_and_Bases{2};
    sig = Data_and_Bases{3};
    
    X_bc = [0; 1];
    
    % PIELM input layer parameters
    m = 1./(sqrt(2) * sig);
    alpha = -m .* alpha_star;
    
    % Design Matrix
    [H, b] = CONSTRUCT_H_AND_b(X_pde, X_bc, m, alpha, nu);
    
    % Least square solution
    c = pinv(H) * b;  % Least-squares solution
    
    % Create validation dataset
    X_val = create_validation_set(X_pde, 0.2, true); % 20% validation set, unique points

    % plot_validation_distribution(X_pde, X_val, 'FIG_VALIDATION_DISTRIBUTION.png');

    % Compute the loss (L-infty norm of residuals)
    [H, b] = CONSTRUCT_H_AND_b(X_val, X_bc, m, alpha, nu);
    J = norm(abs(H * c - b), Inf); 
    J = log10(J);
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
    sig_max = 10*(2/n_unif);
    sig_unif = sig_max*ones(size(alpha_star_unif));

    %% Local Part (Adaptive)
    eta_tau = eta * w_tau/3;
    n_adap = round(0.35*n_unif .* w_f ./ sum(w_f));% MAXIMUM 35%!

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

        % LHS_PDE(k, :) = -(2*(2*X_k-1)).*2 * Phi .* (m' .* (X_k * m' + alpha')) ...
        %                 + 2 * nu * Phi .* (m'.^2) .* (1 - 2 * (X_k * m' + alpha').^2)...
        %                 + 4*Phi;

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
%CREATE_VALIDATION_SET Creates a validation set preserving the distribution of X_pde.
%   X_val = create_validation_set(X_pde, validation_ratio, ensure_unique)
%
% Inputs:
%   - X_pde: Column vector (Nx1) of coordinates in [0,1].
%   - validation_ratio: Fraction of X_pde to include in validation set (default: 0.2).
%   - ensure_unique: If true, removes duplicates in X_val (default: true).
%
% Output:
%   - X_val: Validation set with same distribution as X_pde.

% Set defaults
if nargin < 2
    validation_ratio = 0.2; % 20% validation set by default
end
if nargin < 3
    ensure_unique = true; % Ensure unique points by default
end

% Step 1: Estimate probability density (Kernel Density Estimation)
[pdf_values, ~] = ksdensity(X_pde, X_pde, 'Function', 'pdf');

% Step 2: Compute sampling weights (higher weight for sparser regions)
weights = 1 ./ (pdf_values + eps); % eps avoids division by zero
weights = weights / sum(weights);  % Normalize to probabilities

% Step 3: Sample validation points
n_val = round(validation_ratio * length(X_pde));
val_indices = randsample(1:length(X_pde), n_val, true, weights);
X_val = X_pde(val_indices);

% Step 4: Ensure uniqueness (optional)
if ensure_unique
    X_val = unique(X_val);
end

end


function plot_validation_distribution(X_pde, X_val, save_path)
% PLOT_VALIDATION_DISTRIBUTION Creates publication-quality histograms.
%   plot_validation_distribution(X_pde, X_val, save_path)
%
% Inputs:
%   - X_pde: Original column vector (Nx1).
%   - X_val: Validation set column vector (Mx1).
%   - save_path: Path to save the figure (e.g., 'figures/validation_dist.png').

% Create figure with white background
fig = figure('Units', 'inches', 'Position', [0 0 8 6], 'Color', 'w'); % 8x6 inches
set(fig, 'DefaultAxesFontSize', 14);

% Subplot 1: Original X_pde
subplot(2, 1, 1);
h1 = histogram(X_pde, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');
title('Original $\mathbf{X}_{\mathrm{pde}}$ Density', 'Interpreter', 'LaTeX', 'FontSize', 16);
xlabel('$\mu$', 'Interpreter', 'LaTeX', 'FontSize', 14);
ylabel('PDF', 'Interpreter', 'LaTeX', 'FontSize', 14);
grid on;
box on;

% Subplot 2: Validation set X_val
subplot(2, 1, 2);
h2 = histogram(X_val, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'none');
title('Validation Set $\mathbf{X}_{\mathrm{val}}$ Density', 'Interpreter', 'LaTeX', 'FontSize', 16);
xlabel('$\mu$', 'Interpreter', 'LaTeX', 'FontSize', 14);
ylabel('PDF', 'Interpreter', 'LaTeX', 'FontSize', 14);
grid on;
box on;

% Adjust subplot spacing
set(gcf, 'Units', 'inches', 'Position', [0 0 8 6]); % Width=8", Height=6"
set(gcf, 'PaperPositionMode', 'auto');

% Save as 300 DPI PNG
if nargin > 2
    print(save_path, '-dpng', '-r300');
    disp(['Figure saved to: ' save_path]);
end
end



function stop = stopIfNoImprovement(results, state)
    persistent logLossHistory
    persistent lastChangeIter

    stop = false;

    % Initialize on first call
    if isempty(logLossHistory)
        logLossHistory = [];
        lastChangeIter = 1;
    end

    % Get current loss (already log10-scaled)
    currentLoss = min(results.ObjectiveTrace);
    logLossHistory = [logLossHistory; currentLoss];

    % Check for log-scale improvement of at least 1
    if numel(logLossHistory) > 1
        if (logLossHistory(end-1) - logLossHistory(end)) >= 1
            lastChangeIter = numel(logLossHistory);
        end
    end

    % Stopping Condition 1: No significant improvement in 100 iterations
    if (numel(logLossHistory) - lastChangeIter) >= 100
        stop = true;
        disp('Stopping optimization: log-scaled loss has not improved by ≥1 in the last 100 iterations.');
    end

    % Stopping Condition 2: Loss has reached ≤ -8
    if currentLoss <= -8
        stop = true;
        disp('Stopping optimization: log-scaled loss ≤ -8.');
    end
end



function KAPI_Training_History(results)
    % Extract loss and parameter traces
    loss_history = results.ObjectiveTrace;
    vars = results.XTrace;
    w_names = vars.Properties.VariableNames;
    w_vals = table2array(vars);

    % Initialize tracking arrays
    best_loss_history = zeros(size(loss_history));
    best_params = zeros(size(w_vals));
    min_val = Inf;
    min_idx = 0;

    % Loop through all iterations
    for i = 1:length(loss_history)
        if loss_history(i) < min_val
            min_val = loss_history(i);
            min_idx = i;
        end
        best_loss_history(i) = min_val;
        best_params(i, :) = w_vals(min_idx, :);
    end

    % Plot Best Loss History (log scale)
    figure('Position', [100, 100, 1000, 600]);
    subplot(2,1,1);
    plot(best_loss_history, 'LineWidth', 2, 'Color', 'b');
    xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$log_{10}(J)$', 'Interpreter', 'latex', 'FontSize', 20);
    title('KAPI-ELM Training History', 'Interpreter', 'latex', 'FontSize', 22);
    set(gca, 'FontSize', 20); grid on; box on;

    % Plot parameter evolution (semilogy if needed)
    subplot(2,1,2);
    semilogy(best_params, 'LineWidth', 2);
    xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('Parameter Values', 'Interpreter', 'latex', 'FontSize', 20);
    % legend(w_names, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'eastoutside');
    legend({'$f$', '$\mu$', '$\tau$', '$\lambda$'}, ...
    'Interpreter', 'latex', 'FontSize', 16, 'Location', 'eastoutside');
    set(gca, 'FontSize', 18); grid on; box on;

    % Save figure
    folder = './';
    exportgraphics(gcf, fullfile(folder, 'KAPI_Training_History.png'), 'Resolution', 300);
end
