clc; clear; close all;
rng(42);

%% ===================== 1. Generate synthetic sensor data =====================
% True parameter (unknown in inversion)
nu_true = 0.05;

% Grid used by Poisson_Exact (must match its internal N)
N_exact = 100;
x = linspace(0,1,N_exact);
y = linspace(0,1,N_exact);
[Xg, Yg] = meshgrid(x, y);

% Exact "truth" field from FD solver
u_exact_grid = Poisson_Exact(nu_true);   % N_exact x N_exact

% Choose sensor locations as a coarse subgrid
step = 5;  % every 5th point in each direction
x_sens_mat = Xg(1:step:end, 1:step:end);
y_sens_mat = Yg(1:step:end, 1:step:end);
X_sens = [x_sens_mat(:), y_sens_mat(:)];      % Ns x 2

u_exact_sens = u_exact_grid(1:step:end, 1:step:end);
u_exact_sens = u_exact_sens(:);              % Ns x 1

% Add noise to sensors
noise_level = 0.01*5;
u_obs = u_exact_sens + noise_level * std(u_exact_sens) * randn(size(u_exact_sens));

% Quick sanity check plot (saved as sensor_vs_exact_2D.png)
% plot_sensor_vs_exact_2D(x, y, u_exact_grid, X_sens, u_obs, nu_true, ...
%                         'sensor_vs_exact_2D');
pause(1)
plot_sensor_only_2D(X_sens, u_obs, 'sensor_only_2D');


%% ===================== 2. BO variables (w + nu) ============================
n_unif  = 40;     % same as forward solver
sig_max = 0.2;    % same as forward solver

% KAPI distributional hyperparameter bounds
f_adap_range = [0.5, 1.0];
mu_x_range   = [0.2, 0.8];
mu_y_range   = [0.2, 0.8];
tau_range    = [0.1, 0.5];
lam_range    = [0.5, 0.9];

% Bounds for physical parameter nu
nu_min = 0.005;nu_max = 0.5;

vars = [
    optimizableVariable('f_adap', f_adap_range, 'Type','real')
    optimizableVariable('mu_x',   mu_x_range,   'Type','real')
    optimizableVariable('mu_y',   mu_y_range,   'Type','real')
    optimizableVariable('tau',    tau_range,    'Type','real')
    optimizableVariable('lam',    lam_range,    'Type','real')
    optimizableVariable('nu',    [nu_min,nu_max], 'Transform','log')
];

% Objective wrapper: inverse problem
objectiveFcn = @(p) ObjectiveFcn_INV( ...
    p, X_sens, u_obs, n_unif, sig_max);

%% ===================== 3. Run Bayesian optimization =======================
tic;
results = bayesopt(objectiveFcn, vars, ...
    'MaxObjectiveEvaluations', 150, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'PlotFcn', {@plotMinObjective, @plotObjectiveModel}, ...
    'OutputFcn', @stopIfNoImprovement_INV);
toc;

% Extract and display best parameters
best_struct = results.XAtMinObjective;
best_w = [best_struct.f_adap; ...
          best_struct.mu_x; ...
          best_struct.mu_y; ...
          best_struct.tau; ...
          best_struct.lam];
best_nu = best_struct.nu;

fprintf('\nBest parameters (w, nu):\n');
disp(best_w);
fprintf('Recovered nu: %.4g (true nu = %.4g)\n', best_nu, nu_true);
fprintf('Min log10(J): %.6f\n', results.MinObjective);

% Best-so-far history with nu trace and nu_true reference
pause(5)
KAPI_Training_History_INV_Poisson2D(results);

%% ===================== 4. Reconstruct field with best parameters ===========
% Rebuild collocation/bases using best (w, nu)
[X_f, X_lft, X_ryt, X_bot, X_top, alpha_f, beta_f, m_f, n_f] = ...
    Fixed_PIELM(n_unif, sig_max);

[X_adap, alpha_adap, beta_adap, m_adap, n_adap] = ...
    Adaptive_PIELM(best_nu, n_unif, sig_max, best_w);

X_pde_best = [X_f; X_adap];
m_best     = [m_f, m_adap];
n_best     = [n_f, n_adap];
alpha_best = [alpha_f, alpha_adap];
beta_best  = [beta_f, beta_adap];

% Build training system: PDE + BC + sensor rows
[H_tr, b_tr] = CONSTRUCT_H_AND_b_INV(X_pde_best, ...
    X_lft, X_ryt, X_bot, X_top, ...
    X_sens, m_best, n_best, alpha_best, beta_best, best_nu, u_obs);

% Solve for coefficients
c_best = pinv(H_tr) * b_tr;

% Evaluate PIELM solution on same grid as Poisson_Exact for visualization
X_eval = [Xg(:), Yg(:)];   % N_exact^2 x 2
Phi_eval = RBF_PHI_2D(X_eval, m_best, n_best, alpha_best, beta_best);
u_hat_flat = Phi_eval * c_best;
u_hat_grid = reshape(u_hat_flat, size(Xg));

% Plot reconstruction vs exact, with sensors
pause(1)
plot_reconstruction_2D(x, y, u_exact_grid, u_hat_grid, ...
                       X_sens, u_obs, nu_true, best_nu, ...
                       'reconstruction_2D');

%% ========================= LOCAL FUNCTIONS =================================
%--------------------------------------------------------------------------%
function J = ObjectiveFcn_INV(w_struct, X_sens, u_obs, n_unif, sig_max)
% Inverse objective:
%  1) Build PIELM bases for given (w, nu)
%  2) Construct PDE+BC+sensor system
%  3) Solve for c
%  4) Return log10 of infinity norm of residual on that system

    % Unpack parameters
    w = [w_struct.f_adap; ...
         w_struct.mu_x; ...
         w_struct.mu_y; ...
         w_struct.tau; ...
         w_struct.lam];
    nu = w_struct.nu;

    % Step 1: Uniform distribution + boundary
    [X_f, X_lft, X_ryt, X_bot, X_top, alpha_f, beta_f, m_f, n_f] = ...
        Fixed_PIELM(n_unif, sig_max);

    % Step 2: Adaptive distribution based on (w, nu)
    [X_adap, alpha_adap, beta_adap, m_adap, n_adap] = ...
        Adaptive_PIELM(nu, n_unif, sig_max, w);

    % Step 3: Combine PDE collocation and basis params
    X_pde = [X_f; X_adap];
    m     = [m_f, m_adap];
    n     = [n_f, n_adap];
    alpha = [alpha_f, alpha_adap];
    beta  = [beta_f, beta_adap];

    % Step 4: Build system including sensors
    [H, b] = CONSTRUCT_H_AND_b_INV(X_pde, X_lft, X_ryt, X_bot, X_top, ...
                                   X_sens, m, n, alpha, beta, nu, u_obs);

    % Step 5: Solve and compute residual
    c = pinv(H) * b;
    res = H * c - b;
    J = log10(max(norm(res, Inf), realmin));
end

%--------------------------------------------------------------------------%
function [X_pde, X_lft, X_ryt, X_bot, X_top, alpha, beta, m, n] = ...
         Fixed_PIELM(n_unif, sig_max)
    % Domain
    xL = 0; xR = 1; 
    yB = 0; yT = 1;

    % Uniform grid (cell-centered)
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

    % RBF centers (uniform) and widths
    [alpha_star_unif, beta_star_unif] = meshgrid( ...
        linspace(0, 1, round(0.5 * n_unif)), ...
        linspace(0, 1, round(0.5 * n_unif)));
    alpha_star = alpha_star_unif(:);
    beta_star  = beta_star_unif(:);
    sig_range  = sig_max * ones(size(alpha_star, 1), 1);

    % RBF parameters
    m = 1 ./ (sqrt(2) * sig_range)';
    n = 1 ./ (sqrt(2) * sig_range)';
    alpha = -m .* alpha_star';
    beta  = -n .* beta_star';
end

%--------------------------------------------------------------------------%
function [X_adap, alpha, beta, m, n] = Adaptive_PIELM(nu, n_unif, sig_max, w)
    f_adap = w(1);
    mu_x   = w(2);
    mu_y   = w(3);
    tau    = w(4);
    lam    = w(5);

    eta = 0.1;

    n_adap = round(f_adap * (0.5 * n_unif)^2);

    % Sampling points (collocation)
    X_adap = [mu_x, mu_y] + eta * tau * randn(n_adap, 2);

    % RBF centers
    alpha_star = mu_x + eta * tau * randn(n_adap, 1);
    beta_star  = mu_y + eta * tau * randn(n_adap, 1);

    % RBF widths (nu-dependent)
    inv_sig_ref   = 1 / (sqrt(2) * sig_max);
    inv_sig_nu    = inv_sig_ref * exp((1 + lam) * -log10(nu));
    inv_sig_range = unifrnd(-inv_sig_nu/2, inv_sig_nu/2, 1, n_adap);
    sig_range     = 1 ./ (abs(sqrt(2) * inv_sig_range)' + 1e-5); 

    m     = 1 ./ (sqrt(2) * sig_range)';
    n     = 1 ./ (sqrt(2) * sig_range)';
    alpha = -m .* alpha_star';
    beta  = -n .* beta_star';
end

%--------------------------------------------------------------------------%
function [H, b] = CONSTRUCT_H_AND_b_INV(X_pde, X_bc_left, X_bc_right, ...
                                        X_bc_bottom, X_bc_top, ...
                                        X_sens, m, n, alpha, beta, nu, u_obs)
% Build system:
% [ PDE rows at X_pde
%   BC rows at X_bc
%   Sensor rows at X_sens ]
% with RHS = [f(x); 0; u_obs].

    X_bc = [X_bc_left; X_bc_right; X_bc_bottom; X_bc_top];    
    N_pde = size(X_pde,1);
    N_bc  = size(X_bc,1);
    N_s   = size(X_sens,1);
    NN    = length(alpha);

    x0 = 0.5; y0 = 0.5;   % source center

    % ------ PDE residuals: -Delta u = f(x) ------ %
    LHS_PDE = zeros(N_pde, NN);
    RHS_PDE = zeros(N_pde, 1);
    for k = 1:N_pde
        X_k = X_pde(k, :);
        z1  = m * X_k(1) + alpha;
        z2  = n * X_k(2) + beta;
        z_sqr = z1.^2 + z2.^2;

        % Laplacian of Gaussian RBFs:
        % u_xx + u_yy = 2*m^2*(1 - 2*z1^2)*phi + 2*n^2*(1 - 2*z2^2)*phi
        phi = exp(-z_sqr);
        LHS_PDE(k, :) = -2 * phi .* ( m.^2 .* (1 - 2*z1.^2) + ...
                                       n.^2 .* (1 - 2*z2.^2) );

        % Gaussian RHS
        RHS_PDE(k) = (1/(2*pi*nu^2)) * exp(-((X_k(1) - x0)^2 + (X_k(2) - y0)^2) ...
                                            /(2*nu^2));
    end

    % ------ BC rows (Dirichlet: u=0 on boundary) ------ %
    LHS_BC = zeros(N_bc, NN);
    RHS_BC = zeros(N_bc, 1);
    for k = 1:N_bc
        X_k = X_bc(k, :);
        z1  = m * X_k(1) + alpha;
        z2  = n * X_k(2) + beta;
        z_sqr = z1.^2 + z2.^2;
        LHS_BC(k,:) = exp(-z_sqr);
    end

    % ------ Sensor rows: u(x_sens) ≈ u_obs ------ %
    LHS_SENS = zeros(N_s, NN);
    RHS_SENS = u_obs(:);
    for k = 1:N_s
        X_k = X_sens(k, :);
        z1  = m * X_k(1) + alpha;
        z2  = n * X_k(2) + beta;
        z_sqr = z1.^2 + z2.^2;
        LHS_SENS(k,:) = exp(-z_sqr);
    end

    % Combine
    H = [LHS_PDE; LHS_BC; LHS_SENS];
    b = [RHS_PDE; RHS_BC; RHS_SENS];
end

%--------------------------------------------------------------------------%
function Phi = RBF_PHI_2D(X_eval, m, n, alpha, beta)
% X_eval: (N x 2), m,n,alpha,beta: row vectors (1 x NN)
    if iscolumn(m),     m = m';     end
    if iscolumn(n),     n = n';     end
    if iscolumn(alpha), alpha = alpha'; end
    if iscolumn(beta),  beta = beta';  end

    x = X_eval(:,1);
    y = X_eval(:,2);

    z1 = x * m + alpha;   % N x NN
    z2 = y * n + beta;    % N x NN
    Phi = exp(-(z1.^2 + z2.^2));
end

%--------------------------------------------------------------------------%
function KAPI_Training_History_INV_Poisson2D(results)
% Best-so-far history for 2D Poisson inverse:
% Top: log10(J) best-so-far
% Bottom: best-so-far parameters f,mu_x,mu_y,tau,lambda,nu (nu on log scale)
    % Try to get nu_true from base workspace
    try
        if evalin('base','exist(''nu_true'',''var'')')
            nu_true = evalin('base','nu_true');
        else
            nu_true = [];
        end
    catch
        nu_true = [];
    end

    loss = results.ObjectiveTrace(:);
    W    = table2array(results.XTrace);   % cols: [f_adap mu_x mu_y tau lam nu]
    n    = numel(loss);

    % Best-so-far (running argmin)
    best_loss   = zeros(n,1);
    best_params = zeros(n, size(W,2));
    min_val = inf; min_idx = 1;
    for i = 1:n
        if loss(i) < min_val
            min_val = loss(i); min_idx = i;
        end
        best_loss(i)      = min_val;
        best_params(i, :) = W(min_idx, :);
    end

    % final best params (for annotation)
    bf   = best_params(end,1);
    bmx  = best_params(end,2);
    bmy  = best_params(end,3);
    btau = best_params(end,4);
    blam = best_params(end,5);
    bnu  = best_params(end,6); %#ok<NASGU>

    % Style
    set(0,'DefaultTextInterpreter','latex');
    set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter','latex');
    set(0,'DefaultAxesFontName','Times');
    set(0,'DefaultTextFontName','Times');

    col.blue   = [  0, 114, 178]/255;
    col.vermil = [213,  94,   0]/255;
    col.green  = [  0, 158, 115]/255;
    col.purple = [204, 121, 167]/255;
    col.gold   = [230, 159,   0]/255;
    col.black  = [  0,   0,   0]/255;

    lw = 2.2; lwBold = 2.8; fsAx = 16; fsLb = 18;

    fig = figure('Color','w','Position',[80 80 1100 700]);
    tl  = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

    % Top: best-so-far loss
    ax1 = nexttile(tl,1);
    plot(ax1, best_loss,'-','Color',col.blue,'LineWidth',lwBold);
    grid(ax1,'on'); ax1.GridAlpha = 0.25; ax1.LineWidth = 1.1; ax1.FontSize = fsAx;
    xlabel(ax1,'Iteration','FontSize',fsLb);
    ylabel(ax1,'$\log_{10}(J)$','FontSize',fsLb);
    xlim(ax1,[1 n]);

    % Bottom: best-so-far params
    ax2 = nexttile(tl,2); hold(ax2,'on'); ax2.LineWidth = 1.1; ax2.FontSize = fsAx;

    % Left axis: f, mu_x, mu_y, tau, lambda
    yyaxis(ax2,'left');
    p1 = plot(ax2,best_params(:,1),'-','LineWidth',lw,'Color',col.blue);
    p2 = plot(ax2,best_params(:,2),'-','LineWidth',lw,'Color',col.gold);
    p3 = plot(ax2,best_params(:,3),'-','LineWidth',lw,'Color',col.green);
    p4 = plot(ax2,best_params(:,4),'-','LineWidth',lw,'Color',col.purple);
    p5 = plot(ax2,best_params(:,5),'-','LineWidth',lw,'Color',[0.5 0.5 0.5]);
    ylim(ax2,[0 1.1]);
    ylabel(ax2,'$f,\,\mu_x,\,\mu_y,\,\tau,\,\lambda$','FontSize',fsLb);
    grid(ax2,'on'); ax2.GridAlpha = 0.25; ax2.YColor=[0 0 0];

    % Right axis: nu (log scale)
    yyaxis(ax2,'right');
    p6 = semilogy(ax2,best_params(:,6),'-','LineWidth',lwBold,'Color',col.black, ...
                  'DisplayName','$\nu$');
    if ~isempty(nu_true)
        pref = yline(ax2,nu_true,'--','LineWidth',2,'Color',col.vermil, ...
                     'DisplayName','True $\nu$');
        xlab = round(0.72*n);
        text(ax2,xlab,nu_true, ...
            sprintf('$\\nu_{\\mathrm{true}}=%.2g$',nu_true), ...
            'Interpreter','latex','Color',col.vermil,'FontWeight','bold', ...
            'HorizontalAlignment','left','VerticalAlignment','bottom');
    else
        pref = []; %#ok<NASGU>
    end
    ylabel(ax2,'$\nu$ (log scale)','FontSize',fsLb);
    set(ax2,'YScale','log');
    xlim(ax2,[1 n]);
    xlabel(ax2,'Iteration','FontSize',fsLb);

    if ~isempty(nu_true)
        lg = legend([p1 p2 p3 p4 p5 p6 pref], ...
            {'$f$','$\mu_x$','$\mu_y$','$\tau$','$\lambda$','$\nu$','True $\nu$'}, ...
            'Location','eastoutside','FontSize',13);
    else
        lg = legend([p1 p2 p3 p4 p5 p6], ...
            {'$f$','$\mu_x$','$\mu_y$','$\tau$','$\lambda$','$\nu$'}, ...
            'Location','eastoutside','FontSize',13);
    end
    lg.Box = 'off';

    exportgraphics(fig,'KAPI_Training_History_INV_Poisson2D.png','Resolution',300);
end

%--------------------------------------------------------------------------%
function stop = stopIfNoImprovement_INV(results, state)
% Early stopping for inverse problem: stop if
%  - no improvement >= 1 in log10(J) over last 100 iterations
%  - or best log10(J) <= -5
    persistent logLossHistory
    persistent lastChangeIter

    stop = false;

    if strcmp(state,'start')
        logLossHistory = [];
        lastChangeIter = 1;
        return;
    elseif strcmp(state,'iteration')
        currentLoss = min(results.ObjectiveTrace); % log10-scaled
        logLossHistory = [logLossHistory; currentLoss];

        if numel(logLossHistory) > 1
            if (logLossHistory(end-1) - logLossHistory(end)) >= 1
                lastChangeIter = numel(logLossHistory);
            end
        end

        if (numel(logLossHistory) - lastChangeIter) >= 100
            stop = true;
            disp('Stopping: No log-loss improvement ≥1 in 100 iterations.');
        end

        if currentLoss <= -5
            stop = true;
            disp('Stopping: Log-loss ≤ -5.');
        end
    elseif strcmp(state,'done')
        % reset
        logLossHistory = [];
        lastChangeIter = 1;
    end
end

%--------------------------------------------------------------------------%
function u = Poisson_Exact(nu)
% Same as your forward FD solver; unchanged.
    xL = 0; xR = 1;
    yB = 0; yT = 1;

    N  = 100;
    x0 = 0.5; y0 = 0.5;

    x = linspace(xL, xR, N);
    y = linspace(yB, yT, N);
    [X, Y] = meshgrid(x, y);
    dx = x(2) - x(1);

    f = (1/(2*pi*nu^2))*exp(-((X-x0).^2 + (Y-y0).^2)/(2*nu^2));

    u = zeros(N, N);

    e = ones(N-2, 1);
    D2 = spdiags([e -2*e e], -1:1, N-2, N-2)/dx^2;
    I = speye(N-2);
    A = kron(I, D2) + kron(D2, I);

    f_inner = f(2:end-1, 2:end-1);
    rhs = f_inner(:);

    u_inner = A \ rhs;

    u(2:end-1, 2:end-1) = reshape(u_inner, [N-2, N-2]);
end

%--------------------------------------------------------------------------%
% function plot_sensor_vs_exact_2D(x, y, u_exact_grid, X_sens, u_obs, nu_true, filename)
% % Plot exact field and sensor locations with noisy values.
%     if nargin < 7
%         filename = 'sensor_vs_exact_2D';
%     end
% 
%     figure('Color','w','Position',[100 100 1000 450]);
% 
%     subplot(1,2,1);
%     imagesc(x,y,u_exact_grid);
%     set(gca,'YDir','normal');
%     colorbar;
%     hold on;
%     plot(X_sens(:,1), X_sens(:,2), 'ko', 'MarkerSize',6,'LineWidth',1.5);
%     title(sprintf('Exact u(x,y), \\nu_{true}=%.3g',nu_true), ...
%           'Interpreter','latex','FontSize',16);
%     xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
% 
%     subplot(1,2,2);
%     scatter(X_sens(:,1), X_sens(:,2), 40, u_obs, 'filled');
%     colormap(gca, parula); colorbar;
%     title('Noisy sensor data','Interpreter','latex','FontSize',16);
%     xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
%     set(gca,'FontSize',12);
% 
%     exportgraphics(gcf,[filename '.png'],'Resolution',300);
% end
function plot_sensor_only_2D(X_sens, u_obs, filename)
% Plot ONLY noisy sensor data in 2D
% X_sens : (Ns x 2)
% u_obs  : (Ns x 1)

    if nargin < 3
        filename = 'sensor_only_2D';
    end

    figure('Color','w','Position',[100 100 550 500]);

    % Scatter the sensors colored by observed values
    scatter(X_sens(:,1), X_sens(:,2), 50, u_obs, 'filled');
    colormap(parula); 
    colorbar;

    title('Noisy sensor data','Interpreter','latex','FontSize',18);
    xlabel('$x$','Interpreter','latex','FontSize',20);
    ylabel('$y$','Interpreter','latex','FontSize',20);

    set(gca,'FontSize',16,'LineWidth',1.2,'Box','on');
    axis equal tight;

    exportgraphics(gcf,[filename '.png'],'Resolution',300);
end



%--------------------------------------------------------------------------%
function plot_reconstruction_2D(x, y, u_exact_grid, u_hat_grid, ...
                                X_sens, u_obs, nu_true, nu_rec, filename)
% Plot exact vs reconstructed field and sensor locations.
    if nargin < 9
        filename = 'reconstruction_2D';
    end

    figure('Color','w','Position',[100 100 1200 500]);

    % subplot(1,2,1);
    % imagesc(x,y,u_exact_grid);
    % set(gca,'YDir','normal');
    % colorbar;
    % hold on;
    % plot(X_sens(:,1), X_sens(:,2), 'ko', 'MarkerSize',5,'LineWidth',1.2);
    % title(sprintf('Exact, \\nu_{true}=%.3g', nu_true), ...
    %       'Interpreter','latex','FontSize',16);
    % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
    % 
    % subplot(1,2,2);
    % imagesc(x,y,u_hat_grid);
    % set(gca,'YDir','normal');
    % colorbar;
    % hold on;
    % scatter(X_sens(:,1), X_sens(:,2), 25, u_obs, 'o', 'filled', ...
    %         'MarkerEdgeColor','k');
    % title(sprintf('KAPI-ELM, \\nu_{rec}=%.3g', nu_rec), ...
    %       'Interpreter','latex','FontSize',16);
    % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex');
    subplot(1,2,1);
imagesc(x,y,u_exact_grid);
set(gca,'YDir','normal');
colorbar;
hold on;
plot(X_sens(:,1), X_sens(:,2), 'ko', 'MarkerSize',5,'LineWidth',1.2);
title(sprintf('$\\nu_{\\mathrm{true}}=%.3g$', nu_true), ...
      'Interpreter','latex','FontSize',16);

subplot(1,2,2);
imagesc(x,y,u_hat_grid);
set(gca,'YDir','normal');
colorbar;
hold on;
scatter(X_sens(:,1), X_sens(:,2), 25, u_obs, 'o','filled','MarkerEdgeColor','k');
title(sprintf('$\\nu_{\\mathrm{rec}}=%.3g$', nu_rec), ...
      'Interpreter','latex','FontSize',16);


    sgtitle('2D Poisson inverse reconstruction','Interpreter','latex','FontSize',18);
    exportgraphics(gcf,[filename '.png'],'Resolution',300);
end
