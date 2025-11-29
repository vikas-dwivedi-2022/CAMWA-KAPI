% KAPI-ELM Inverse (Deterministic): Identify scalar nu with Joint BO over (w_f, w_mu, w_tau, w_lam, nu)
% Data mismatch is handled like Dirichlet BC rows.
% Produces:
%   - KAPI_Training_History.png  (best-so-far loss & params with true-nu reference)
%   - sensor_vs_exact.png        (sensors vs exact, quick check)
%   - sensor_vs_exact_and_pred.png (exact vs prediction vs sensors using optimized params)
clc; clear; close all;
rng(42);

%% -------------------- Generate sensor data (like BC rows) --------------------
nu_true = 5e-3;                    % ground-truth (for data generation only)
x_sens  = shishkin_mesh()';        % sensor locations in [0,1]
u_exact = EXACT_SOLUTION(x_sens, nu_true);

noise_level = 0.05;                % e.g., 0.00, 0.01, 0.05
u_obs = u_exact + noise_level*std(u_exact)*randn(size(u_exact));

% Quick visual (saved as sensor_vs_exact.png)
plot_sensor_vs_exact(x_sens, u_obs, u_exact, nu_true, 'sensor_vs_exact');

%% -------------------- BO variables & bounds (deterministic) ------------------
eta    = 0.1;         % controls adaptive cluster spread
n_unif = 1500;        % baseline collocation count

% Bounds for KAPI distributional hyperparameters
w_f_range   = [0.495, 0.505];
w_mu_range  = [0.90,  0.99];
w_tau_range = [0.10,  0.40];
w_lam_range = [0.50,  1.00];

% Bound for the PHYSICAL PARAMETER nu (deterministic scalar)
nu_min = 1e-4; 
nu_max = 1e-2;        % widen if needed

% Pack BO variables
optimVars = [
    optimizableVariable('w1', w_f_range)                          % f
    optimizableVariable('w2', w_mu_range)                         % mu
    optimizableVariable('w3', w_tau_range)                        % tau
    optimizableVariable('w4', w_lam_range)                        % lambda
    optimizableVariable('nu', [nu_min, nu_max], 'Transform','log')% scalar nu
];

% Objective: PDE residual rows + sensor rows; validation like forward
objectiveFcn = @(p) KAPI_INV_objective( ...
    p.nu, eta, n_unif, ...
    p.w1, p.w2, p.w3, p.w4, ...
    x_sens, u_obs);

%% -------------------- Run Bayesian optimization ------------------------------
tic;
results = bayesopt( ...
    objectiveFcn, optimVars, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 50, ...
    'IsObjectiveDeterministic', true, ... % deterministic: RNG fixed inside generator
    'UseParallel', false, ...
    'Verbose', 0, ...
    'OutputFcn', @stopIfNoImprovement);
toc;

best = results.XAtMinObjective;
fprintf('\nBest parameters:\n'); disp(best);
fprintf('Min log10(J): %.6f\n', results.MinObjective);

% Best-so-far history (figure saved as KAPI_Training_History.png)
KAPI_Training_History(results);

%% -------------------- Reconstruct with best parameters -----------------------
% Rebuild bases with best (w, nu)
fixed_best   = struct('nu', best.nu, 'eta', eta, 'n_unif', n_unif);
tunable_best = struct('w_f', best.w1, 'w_mu', best.w2, 'w_tau', best.w3, 'w_lam', best.w4);

Data_best = Data_and_Bases_1D(fixed_best, tunable_best);
X_pde_b   = Data_best{1};
a_star_b  = Data_best{2};
sig_b     = Data_best{3};

m_b     = 1./(sqrt(2)*sig_b);
alpha_b = -m_b .* a_star_b;

% Train coefficients on TRAIN set (all collocation + sensor rows)
[H_tr_b, b_tr_b] = CONSTRUCT_H_AND_b(X_pde_b, x_sens, m_b, alpha_b, best.nu, u_obs);
c_b = pinv(H_tr_b) * b_tr_b;

% Evaluate prediction on a dense grid + at sensors
x_plot      = linspace(0,1,600)';       % smooth curve
Phi_plot    = RBF_PHI(x_plot, m_b, alpha_b);
u_hat_plot  = Phi_plot * c_b;

Phi_sens    = RBF_PHI(x_sens, m_b, alpha_b);
u_hat_sens  = Phi_sens * c_b;

% Exact for figure only (not used by inverse)
u_exact_plot = EXACT_SOLUTION(x_plot, nu_true);

% Plot reconstruction figure
pause(2)
plot_reconstruction(x_plot, u_exact_plot, u_hat_plot, ...
                    x_sens, u_obs, u_hat_sens, ...
                    nu_true, best.nu, 'sensor_vs_exact_and_pred');

%% ============================== FUNCTIONS ====================================

function J = KAPI_INV_objective(nu, eta, n_unif, w_f, w_mu, w_tau, w_lam, x_sens, u_obs)
% Deterministic BO objective:
% 1) Build collocation + RBFs with (w, nu)
% 2) Solve analytic LS for c on a training collocation set
% 3) Evaluate residual (PDE + sensor rows) on a validation set
% 4) Return log10 of infinity-norm residual
    fixed_params   = struct('nu', nu, 'eta', eta, 'n_unif', n_unif);
    tunable_params = struct('w_f', w_f, 'w_mu', w_mu, 'w_tau', w_tau, 'w_lam', w_lam);

    % --- Input layer / bases (deterministic given nu,w; RNG fixed inside)
    Data = Data_and_Bases_1D(fixed_params, tunable_params);
    X_pde   = Data{1};
    a_star  = Data{2};
    sig     = Data{3};

    % --- Input-layer parameters for Gaussian features
    m     = 1./(sqrt(2)*sig);
    alpha = -m .* a_star;

    % --- TRAIN: LS on subset of collocation (here: all X_pde) + sensor rows
    [H_tr, b_tr] = CONSTRUCT_H_AND_b(X_pde, x_sens, m, alpha, nu, u_obs);
    c = pinv(H_tr)*b_tr;

    % --- VALIDATE: use (re-weighted) validation collocation + the SAME sensors
    X_val = create_validation_set(X_pde, 0.2, true);  % 20% held-out
    [H_val, b_val] = CONSTRUCT_H_AND_b(X_val, x_sens, m, alpha, nu, u_obs);

    % --- Loss: infinity-norm residual on validation set (as in forward)
    res = norm(H_val*c - b_val, Inf);
    J   = log10(max(res, realmin));
end

function output_cell = Data_and_Bases_1D(fixed_params, tunable_params)
% Deterministic basis generator (RNG fixed), with adaptive widths linked to nu.
    nu     = fixed_params.nu;
    eta    = fixed_params.eta;
    n_unif = fixed_params.n_unif;

    rng(42); % fix RNG -> deterministic objective

    w_f   = tunable_params.w_f;
    w_mu  = tunable_params.w_mu;
    w_tau = tunable_params.w_tau;
    w_lam = tunable_params.w_lam;

    % --- Uniform component
    x_unif          = linspace(0,1,n_unif)';                 % PDE collocation
    alpha_star_unif = linspace(0,1, round(0.5*n_unif))';     % centers
    sig_max         = 10*(2/n_unif);
    sig_unif        = sig_max*ones(size(alpha_star_unif));

    % --- Adaptive component (up to 35% of n_unif)
    eta_tau  = eta * w_tau/3;
    n_adap   = round(0.35*n_unif * w_f / sum(w_f));
    inv_ref  = 1/(sqrt(2)*sig_max);

    x_gauss = []; a_gauss = []; s_gauss = [];
    for i = 1:length(w_mu)
        n_i   = n_adap(i);
        lam_i = w_lam(i);

        % Centers: Gaussian around mu with spread eta_tau, reflected into [0,1]
        raw = w_mu(i) + eta_tau(i)*randn(n_i,1);
        refl = abs(mod(raw,2)); refl = min(refl, 2-refl);

        % Widths: inverse-scale grows with nu^{-(1+lambda)}, bounded by sig_max
        inv_max = inv_ref * exp((1+lam_i)*-log10(nu));  % ~ nu^{-(1+lambda)}
        inv_s   = unifrnd(-inv_max/2, inv_max/2, n_i, 1);
        bw      = min(1./(sqrt(2)*abs(inv_s + eps)), sig_max);

        x_gauss = [x_gauss; refl];
        a_gauss = [a_gauss; raw];
        s_gauss = [s_gauss; bw];
    end

    % Combine
    X_pde   = [x_unif; x_gauss];
    a_star  = [alpha_star_unif; a_gauss];
    sig     = [sig_unif; s_gauss];

    output_cell = {X_pde, a_star, sig};
end

function [H, b] = CONSTRUCT_H_AND_b(X_pde, X_sens, m, alpha, nu, u_obs)
% Build design matrix:
%   [  PDE residual rows at X_pde
%      Sensor/BC rows at X_sens ]
% with RHS b = [0; u_obs].
    Nc  = numel(X_pde);
    Ns  = numel(X_sens);
    NN  = numel(alpha);

    % --- PDE rows (-nu u_xx + u_x = 0)
    LHS_PDE = zeros(Nc, NN);
    for k = 1:Nc
        x = X_pde(k,1);
        z = x*m' + alpha';
        Phi = exp(-(z.^2));
        % u_x   = -2 m z Phi
        % u_xx  =  2 m^2 (1 - 2 z^2) Phi
        LHS_PDE(k,:) = -2*Phi.*(m'.*z) + 2*nu*Phi.*(m'.^2).*(1 - 2*z.^2);
    end

    % --- Sensor rows (just like Dirichlet BC rows)
    LHS_SENS = zeros(Ns, NN);
    for k = 1:Ns
        x = X_sens(k,1);
        z = x*m' + alpha';
        Phi = exp(-(z.^2));
        LHS_SENS(k,:) = Phi;
    end

    H = [LHS_PDE; LHS_SENS];
    b = [zeros(Nc,1); u_obs(:)];
end

function Phi = RBF_PHI(x, m, alpha)
% Build Gaussian RBF design matrix for prediction only (no derivatives).
% x: (Nx1), m, alpha: (1xNN) or (NNx1) vectors
    if iscolumn(m),  m = m';  end
    if iscolumn(alpha), alpha = alpha'; end
    z = x * m + alpha;           % (N x NN)
    Phi = exp(-(z.^2));          % (N x NN)
end

function x = shishkin_mesh()
% Simple two-region Shishkin-style mesh (user-chosen transition)
    N = 50; tau = 0.9;
    if mod(N,2)~=0, error('N must be even.'); end
    N2 = N/2;
    xL = linspace(0, tau, N2+1);
    xR = linspace(tau, 1,   N2+1); xR = xR(2:end);
    x  = [xL, xR];
end

function u_exact = EXACT_SOLUTION(X, nu)
% Stable exact solution: u(x) = (e^{x/nu}-1)/(e^{1/nu}-1), 0<=x<=1
    overflow_threshold = 1/log(realmax);
    if nu > overflow_threshold
        u_exact = expm1(X./nu) ./ expm1(1/nu);
    else
        exponent  = (X - 1)./nu;
        threshold = -log(eps(class(X)));
        u_exact   = exp(exponent);
        u_exact(exponent < -threshold) = 0;
        u_exact(X==1) = 1;
    end
end

function X_val = create_validation_set(X_pde, validation_ratio, ensure_unique)
% Sample a validation subset preserving distribution; deterministic via rng(42) upstream
    if nargin<2, validation_ratio = 0.2; end
    if nargin<3, ensure_unique = true; end
    rng(42);
    [pdf_vals, ~] = ksdensity(X_pde, X_pde, 'Function','pdf');
    w = 1./(pdf_vals + eps); w = w/sum(w);
    n_val = max(1, round(validation_ratio*numel(X_pde)));
    idx = randsample(1:numel(X_pde), n_val, true, w);
    X_val = X_pde(idx);
    if ensure_unique, X_val = unique(X_val); end
end

function stop = stopIfNoImprovement(results, state)
% Early stopping: if best log10(J) hasn't improved by ≥1 in 100 iterations,
% or if best log10(J) <= -8.
    persistent best lastIter
    stop = false;
    switch state
        case 'start'
            best = inf; lastIter = 1;
        case 'iteration'
            k = numel(results.ObjectiveTrace);
            curr = min(results.ObjectiveTrace(1:k));
            if curr < best - 1e-12, best = curr; lastIter = k; end
            if (k - lastIter) >= 100, stop = true; end
            if best <= -8, stop = true; end
        case 'done'
            best = inf; lastIter = 1;
    end
end

function KAPI_Training_History(results)
% Publication-quality best-so-far history for KAPI-ELM inverse.
% Top: best-so-far log10(J). Bottom: best-so-far params; left axis (f,mu,tau,lambda),
% right axis (nu, log scale) with a horizontal reference line for nu_true if available.

    % Try to pull nu_true from base workspace (optional)
    try
        if evalin('base','exist(''nu_true'',''var'')')
            nu_true = evalin('base','nu_true');
        else
            nu_true = [];
        end
    catch
        nu_true = [];
    end

    % ---------- Data ----------
    loss = results.ObjectiveTrace(:);
    W    = table2array(results.XTrace);   % columns: [w1 w2 w3 w4 nu] -> [f mu tau lambda nu]
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

    % Final best values (for annotation)
    bf = best_params(end,1);
    bmu = best_params(end,2);
    btau = best_params(end,3);
    blam = best_params(end,4);
    bnu = best_params(end,5);

    % ---------- Style ----------
    set(0,'DefaultTextInterpreter','latex');
    set(0,'DefaultLegendInterpreter','latex');
    set(0,'DefaultAxesTickLabelInterpreter','latex');
    set(0,'DefaultAxesFontName','Times');
    set(0,'DefaultTextFontName','Times');

    % Colorblind-friendly palette
    col.blue   = [  0, 114, 178]/255;
    col.vermil = [213,  94,   0]/255;
    col.green  = [  0, 158, 115]/255;
    col.purple = [204, 121, 167]/255;
    col.gold   = [230, 159,   0]/255;
    col.black  = [  0,   0,   0]/255;

    lw = 2.2; lwBold = 2.8; fsAx = 18; fsLb = 20;

    % ---------- Figure ----------
    fig = figure('Color','w','Position',[80 80 1100 700]);
    tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

    % ---------- Top: best-so-far loss ----------
    ax1 = nexttile(tl,1);
    plot(ax1,best_loss,'-','Color',col.blue,'LineWidth',lwBold);
    grid(ax1,'on'); ax1.GridAlpha = 0.25; ax1.LineWidth = 1.1; ax1.FontSize = fsAx;
    xlabel(ax1,'Iteration','FontSize',fsLb);
    ylabel(ax1,'$\log_{10}(J)$','FontSize',fsLb);
    xlim(ax1,[1 n]);

    % ---------- Bottom: best-so-far params ----------
    ax2 = nexttile(tl,2); hold(ax2,'on'); ax2.LineWidth = 1.1; ax2.FontSize = fsAx;

    % Left axis (linear): f, mu, tau, lambda
    yyaxis(ax2,'left');
    p1 = plot(ax2,best_params(:,1),'-','LineWidth',lw,'Color',col.blue);   % f
    p2 = plot(ax2,best_params(:,2),'-','LineWidth',lw,'Color',col.gold);   % mu
    p3 = plot(ax2,best_params(:,3),'-','LineWidth',lw,'Color',col.green);  % tau
    p4 = plot(ax2,best_params(:,4),'-','LineWidth',lw,'Color',col.purple); % lambda
    ylim(ax2,[0 1.05]); ylabel(ax2,'$f,\,\mu,\,\tau,\,\lambda$','FontSize',fsLb);
    grid(ax2,'on'); ax2.GridAlpha = 0.25; ax2.YColor = [0 0 0];

    % --- Right axis (log): nu + reference
    yyaxis(ax2,'right');
    p5 = semilogy(ax2,best_params(:,5),'-','LineWidth',lwBold,'Color',col.black, ...
                  'DisplayName','$\nu$');  % ν (black)
    if ~isempty(nu_true)
        pref = yline(ax2,nu_true,'--','LineWidth',2,'Color',col.vermil, ...
                     'DisplayName','True $\nu$');  % legend-ready
        % Optional: place a red label near the right side (no \textcolor)
        xlab = round(0.72*n);  % adjust if needed
        text(ax2, xlab, nu_true, ...
            sprintf('$\\nu_{\\mathrm{true}}=%.2g$', nu_true), ...
            'Interpreter','latex', 'Color', col.vermil, 'FontWeight','bold', ...
            'HorizontalAlignment','left', 'VerticalAlignment','bottom');
    end
    ylabel(ax2,'$\nu$ (log scale)','FontSize',fsLb);
    set(ax2,'YScale','log');
    xlim(ax2,[1 n]);
    xlabel(ax2,'Iteration','FontSize',fsLb);

    % --- Legend (now includes the true-ν line)
    if ~isempty(nu_true)
        lg = legend([p1 p2 p3 p4 p5 pref], ...
            {'$f$','$\mu$','$\tau$','$\lambda$','$\nu$','True $\nu$'}, ...
            'Location','eastoutside','FontSize',14);
    else
        lg = legend([p1 p2 p3 p4 p5], ...
            {'$f$','$\mu$','$\tau$','$\lambda$','$\nu$'}, ...
            'Location','eastoutside','FontSize',14);
    end
    lg.Box = 'off';

    exportgraphics(fig,'KAPI_Training_History.png','Resolution',300);
end

function plot_sensor_vs_exact(x_bc, u_bc, u_bc_exact, nu_true, output_filename)
% Plot sensors vs exact solution (pre-optimization sanity check)
    if nargin<5, output_filename = 'sensor_vs_exact'; end
    figure('Color','w','Position',[100 100 1000 600]); hold on;
    plot(x_bc, u_bc_exact, 'k-', 'LineWidth', 2.5);
    plot(x_bc, u_bc, 'o', 'MarkerSize', 7, 'MarkerEdgeColor', [213 94 0]/255, ...
        'MarkerFaceColor', [213 94 0]/255);
    xlabel('$x$','Interpreter','latex','FontSize',28);
    ylabel('$u(x)$','Interpreter','latex','FontSize',28);
    legend({'Exact $u(x)$','Noisy Sensor Data'},'Interpreter','latex','FontSize',18, ...
           'Location','northwest'); grid on;
    set(gca,'FontSize',22,'LineWidth',1.2,'Box','on');
    exportgraphics(gcf,[output_filename '.png'],'Resolution',300);
end

function plot_reconstruction(x_plot, u_exact_plot, u_hat_plot, ...
                             x_sens, u_obs, u_hat_sens, ...
                             nu_true, nu_rec, filename)
% Plot exact vs predicted field and overlay noisy sensors (+ predicted at sensors)
    if nargin < 9, filename = 'sensor_vs_exact_and_pred'; end

    figure('Color','w','Position',[100 100 1100 600]); hold on;

    % Colors
    col_exact = [0 0 0];                 % black
    col_pred  = [0 114 178]/255;         % blue
    col_sens  = [213 94 0]/255;          % vermilion

    % Curves
    p1 = plot(x_plot, u_exact_plot, '-', 'Color', col_exact, 'LineWidth', 2.5);
    p2 = plot(x_plot, u_hat_plot,  '-', 'Color', col_pred,  'LineWidth', 2.5);

    % Sensors (noisy) and predicted-at-sensors (optional markers)
    p3 = plot(x_sens, u_obs, 'o', 'MarkerSize', 6, 'MarkerEdgeColor', col_sens, ...
              'MarkerFaceColor', col_sens, 'LineStyle','none');
    plot(x_sens, u_hat_sens, 's', 'MarkerSize', 5, 'MarkerEdgeColor', col_pred, ...
         'MarkerFaceColor', 'w', 'LineStyle','none');

    % Labels/legend
    xlabel('$x$', 'Interpreter','latex','FontSize', 28);
    ylabel('$u(x)$', 'Interpreter','latex','FontSize', 28);

    lg = legend([p1 p2 p3], ...
        {'Exact $u(x)$', 'KAPI-ELM (pred.)', 'Noisy sensors'}, ...
        'Interpreter','latex','FontSize', 18, 'Location','northwest');
    lg.Box = 'off';

    % Title annotation (nu_true vs recovered)
    txt = sprintf('Inverse reconstruction: $\\nu_{\\rm true}=%.3g$, $\\nu_{\\rm rec}=%.3g$', ...
                   nu_true, nu_rec);
    title(txt, 'Interpreter','latex','FontSize',20);

    grid on; set(gca,'FontSize', 22, 'LineWidth',1.2, 'Box','on');
    xlim([0 1]);

    exportgraphics(gcf, [filename '.png'], 'Resolution', 300);
end
