clc; clear; close all;
rng(42)
% Define optimizable variables
vars = [
        optimizableVariable('f_adap', [0.5, 1], 'Type', 'real');
        optimizableVariable('mu_x',   [0.2, 0.8],   'Type', 'real');
        optimizableVariable('mu_y',   [0.2, 0.8],   'Type', 'real');
        optimizableVariable('tau',    [0.1, 0.5],   'Type', 'real');
        optimizableVariable('lam',    [0.5, 0.9],   'Type', 'real');
    ];

tic;
results = bayesopt(@ObjectiveFcn, vars, ...
    'MaxObjectiveEvaluations', 300, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotMinObjective, @plotObjectiveModel}, ...
    'OutputFcn', @stopIfNoImprovement);

toc;
% Extract and display best parameters
best_w_struct = results.XAtMinObjective;
best_w = [best_w_struct.f_adap; ...
          best_w_struct.mu_x; ...
          best_w_struct.mu_y; ...
          best_w_struct.tau; ...
          best_w_struct.lam];
    
fprintf('\nBest Parameters Found:\n');
disp(best_w);

% Plot Training History
KAPI_Training_History(results);

%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
function J = ObjectiveFcn(w_struct)
    % Convert struct to parameter vector
    w = [w_struct.f_adap; ...
         w_struct.mu_x; ...
         w_struct.mu_y; ...
         w_struct.tau; ...
         w_struct.lam];

    % Fixed PDE parameter and grid resolution
    nu      = 0.01; %---> WIDTH OF GAUSSIAN SOURCE
    n_unif  = 40;
    sig_max = 0.2;

    % Step 1: Uniform distribution
    [X_f, X_lft, X_ryt, X_bot, X_top, alpha_f, beta_f, m_f, n_f] = Fixed_PIELM(n_unif, sig_max);

    % Step 2: Adaptive distribution based on w
    [X_adap, alpha_adap, beta_adap, m_adap, n_adap] = Adaptive_PIELM(nu, n_unif, sig_max, w);

    % Step 3: Combine fixed and adaptive points
    X_pde = [X_f; X_adap];
    m = [m_f, m_adap];
    n = [n_f, n_adap];
    alpha = [alpha_f, alpha_adap];
    beta  = [beta_f, beta_adap];

    % Step 4: Construct H matrix and b vector
    [H, b] = CONSTRUCT_H_AND_b(X_pde, X_lft, X_ryt, X_bot, X_top, m, n, alpha, beta, nu);

    % Step 5: Solve and compute loss
    c = pinv(H) * b;
    J = norm(H * c - b, Inf); 
    J = log10(J);
end
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
function [X_pde, X_lft, X_ryt, X_bot, X_top, alpha, beta, m, n] = Fixed_PIELM(n_unif, sig_max)
    % Domain boundaries
    xL = 0; xR = 1; 
    yB = 0; yT = 1;

    % Uniform grid
    del = (xR - xL) / n_unif;
    x = linspace(xL + 0.5*del, xR - 0.5*del, n_unif);
    y = linspace(yB + 0.5*del, yT - 0.5*del, n_unif);
    [xx, yy] = meshgrid(x, y);
    X_pde = [xx(:), yy(:)];  
    N_pde = length(X_pde);

    % Boundary points
    N_bdy = round(0.25 * N_pde);
    X_lft = [zeros(N_bdy, 1), linspace(0, 1, N_bdy)'];
    X_ryt = [ones(N_bdy, 1), linspace(0, 1, N_bdy)'];
    X_bot = [linspace(0, 1, N_bdy)', zeros(N_bdy, 1)];
    X_top = [linspace(0, 1, N_bdy)', ones(N_bdy, 1)];

    % RBF centers and widths (0.5*n_unif)
    [alpha_star_unif, beta_star_unif] = meshgrid(linspace(0, 1, round(0.5 * n_unif)), ...
                                                 linspace(0, 1, round(0.5 * n_unif)));
    alpha_star = alpha_star_unif(:);
    beta_star = beta_star_unif(:);
    sig_range = sig_max * ones(size(alpha_star, 1), 1);


    % Compute RBF parameters
    m = 1 ./ (sqrt(2) * sig_range)';
    n = 1 ./ (sqrt(2) * sig_range)';
    alpha = -m .* alpha_star';
    beta  = -n .* beta_star';

end

%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->

function [X_adap, alpha, beta, m, n] = Adaptive_PIELM(nu, n_unif, sig_max, w)

f_adap = w(1);
mu_x   = w(2);
mu_y   = w(3);
tau    = w(4);
lam    = w(5);

%eta=10^floor(log10(1-0)-1);
eta = 0.1;

n_adap = round(f_adap * (0.5*n_unif)^2);

% Sampling Points
X_adap = [mu_x, mu_y] + eta * tau * randn(n_adap, 2);

% RBF centers
alpha_star = mu_x + eta * tau * randn(n_adap, 1);
beta_star  = mu_y + eta * tau * randn(n_adap, 1);

% RBF widths
inv_sig_ref = 1 / (sqrt(2) * sig_max);
inv_sig_nu = inv_sig_ref * exp((1 + lam) * -log10(nu));
inv_sig_range = unifrnd(-inv_sig_nu/2, inv_sig_nu/2, 1, n_adap);
sig_range = 1 ./ (abs(sqrt(2) * inv_sig_range)' + 1e-5); 


m = 1 ./ (sqrt(2) * sig_range)';
n = 1 ./ (sqrt(2) * sig_range)';
alpha = -m .* alpha_star';
beta  = -n .* beta_star';

end

%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
function [H, b] = CONSTRUCT_H_AND_b(X_pde, X_bc_left, X_bc_right, X_bc_bottom, X_bc_top, m, n, alpha, beta,nu)
X_bc = [X_bc_left; X_bc_right; X_bc_bottom; X_bc_top];    
N_pde = length(X_pde);
N_bc = length(X_bc);
NN = length(alpha);
x0 = 0.5; y0 = 0.5;%---> CENTER COORDINATES OF GAUSSIAN SOURCE 

% PDE residuals
LHS_PDE = zeros(N_pde, NN);
RHS_PDE = zeros(N_pde, 1);
for k = 1:N_pde
    X_k = X_pde(k, :);
    z_sqr = (m * X_k(1) + alpha).^2 + (n * X_k(2) + beta).^2;
    LHS_PDE(k, :) = -2 * exp(-z_sqr) .* ((m.^2) .* (1 - 2 * ((m * X_k(1) + alpha).^2)) + ...
                                        (n.^2) .* (1 - 2 * ((n * X_k(2) + beta).^2)));
    RHS_PDE(k) = (1/(2*pi*nu^2))*exp(-((X_k(1) - x0).^2 + (X_k(2) - y0).^2)/(2*nu^2));
end

% BC residuals
LHS_BC = zeros(N_bc, NN);
for k = 1:N_bc
    X_k = X_bc(k, :);
    z_sqr = (m * X_k(1) + alpha).^2 + (n * X_k(2) + beta).^2;
    LHS_BC(k, :) = exp(-z_sqr);
end
RHS_BC = zeros(N_bc, 1);

% Combine system
H = [LHS_PDE; LHS_BC];
b = [RHS_PDE; RHS_BC];
end
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
function KAPI_Training_History(results)
    % Extract traces
    log_loss = results.ObjectiveTrace;
    vars = results.XTrace;
    w_names = vars.Properties.VariableNames;
    w_vals = table2array(vars);

    % Initialize history
    best_log_loss = zeros(size(log_loss));
    best_w = zeros(size(w_vals));
    min_val = Inf;
    min_idx = 0;

    % Track best log-loss and parameters
    for i = 1:length(log_loss)
        if log_loss(i) < min_val
            min_val = log_loss(i);
            min_idx = i;
        end
        best_log_loss(i) = min_val;
        best_w(i, :) = w_vals(min_idx, :);
    end

    % Plot
    figure('Position', [100, 100, 1000, 600]);

    % Log-loss history
    subplot(2,1,1);
    plot(best_log_loss, 'b-', 'LineWidth', 2);
    xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 16);
    % ylabel('log_{10}(Best J)', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel('$\log_{10}(J)$', 'Interpreter', 'latex', 'FontSize', 20);

    % title('KAPI-ELM Training History', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 16); grid on; box on;

    % Parameter evolution
    subplot(2,1,2);
    semilogy(best_w, 'LineWidth', 2);
    xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('Parameter Values', 'Interpreter', 'latex', 'FontSize', 18);
    % legend(w_names, 'Interpreter', 'latex', 'Location', 'eastoutside', 'FontSize', 10);
    w_labels = {'$f$','$\mu_x$', '$\mu_y$', '$\tau$', '$\lambda$' };
    legend(w_labels, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'eastoutside');

    set(gca, 'FontSize', 14); grid on; box on;

    % Save figure
    pause(0.5)
    exportgraphics(gcf, 'FIG_01_KAPI-ELM_Training_History_Poisson_2D.png', 'Resolution', 300);


    % Save CSVs (without nu_str_clean)
    writematrix(best_log_loss(:), 'Best_Loss_History.csv');
    writematrix(best_w, 'Best_W_History.csv');

end
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
function stop = stopIfNoImprovement(results, state)
    persistent logLossHistory
    persistent lastChangeIter

    stop = false;

    % Initialize on first call
    if isempty(logLossHistory) || strcmp(state, 'start')
        logLossHistory = [];
        lastChangeIter = 1;
    end

    % Append current best log-loss
    currentLoss = min(results.ObjectiveTrace); % Already assumed to be log10-scaled
    logLossHistory = [logLossHistory; currentLoss];

    % Check for log-scale improvement
    if numel(logLossHistory) > 1
        if (logLossHistory(end-1) - logLossHistory(end)) >= 1
            lastChangeIter = numel(logLossHistory);
        end
    end

    % Criterion 1: No improvement ≥1 in last 100 iterations
    if (numel(logLossHistory) - lastChangeIter) >= 100
        stop = true;
        disp('Stopping: No log-loss improvement ≥1 in 100 iterations.');
    end

    % Criterion 2: Loss already sufficiently small (e.g., ≤ -8)
    if currentLoss <= -5
        stop = true;
        disp('Stopping: Log-loss ≤ -5.');
    end
end

%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
%-------------------------------------------------------------------------->
