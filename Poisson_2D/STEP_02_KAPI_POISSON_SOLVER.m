clc; clear; close all;
rng(42)
%% INPUT-1: Fixed Parameters
nu = 0.01;         % Viscosity
n_unif = 40;      % Fixed Grid resolution
sig_max = 0.2;    % Maximum RBF width

%% INPUT-2: Tunable Parameters
w = [0.7723;0.4915;0.5068;0.1554;0.8869];

%% STEP-1: Fixed Design Matrix
[X_f, X_lft, X_ryt, X_bot, X_top, alpha_f, beta_f, m_f, n_f ] = Fixed_PIELM(n_unif, sig_max);

%% STEP-2: Tunable Design Matrix
[X_adap, alpha_adap, beta_adap, m_adap, n_adap] = Adaptive_PIELM(nu, n_unif, sig_max, w);

%% STEP-3: Assemble Fixed (Uniform) + Adaptive
X_pde = [X_f;X_adap];
m=[m_f m_adap];
n=[n_f n_adap];
alpha=[alpha_f alpha_adap];
beta=[beta_f beta_adap];

%% STEP-4: Design Matrix and RHS
[H, b] = CONSTRUCT_H_AND_b(X_pde, X_lft, X_ryt, X_bot, X_top, m, n, alpha, beta, nu);

%% STEP-5: Solve and compute residual
c = pinv(H) * b; 
J = norm(H * c - b, Inf);

%% STEP-6: Solve and compute residual
u_Test = Plot_Solution(0,1,0,1,m,n,alpha,beta,c);
u_Exact = Poisson_Exact(nu);
plotComparisonWithErrorContour(u_Test, u_Exact,nu)

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
x0 = 0.5; y0 = 0.5;

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
function u_Test=Plot_Solution(xL,xR,yB,yT,m,n,alpha,beta,c_ref)
N = 100;
x = linspace(xL, xR, N);
y = linspace(yB, yT, N);
[xx, yy] = meshgrid(x, y);
X_test = [xx(:), yy(:)];
N_test=length(X_test);
u_Test=zeros(N_test,1);
for k = 1:N_test
    X_k = X_test(k, :);
    z_sqr = (m * X_k(1) + alpha).^2 + (n * X_k(2) + beta).^2;
    u_Test(k, :) = exp(-z_sqr)*c_ref;
end
% figure;
u_Test=reshape(u_Test, N, N);
% surf(xx, yy,u_Test );  % Actual plotting
% title('PIELM Solution');
% shading interp; colorbar;
end
%------------------------------------------------------>
%------------------------------------------------------>
%------------------------------------------------------>
function u = Poisson_Exact(nu)
% Domain
xL = 0; xR = 1;
yB = 0; yT = 1;

% Parameters
N = 100;                 % Number of grid points
x0 = 0.5; y0 = 0.5;      % Source location

% Create grid
x = linspace(xL, xR, N);
y = linspace(yB, yT, N);
[X, Y] = meshgrid(x, y);
dx = x(2) - x(1);

% 1. Gaussian RHS (normalized)
f = (1/(2*pi*nu^2))*exp(-((X-x0).^2 + (Y-y0).^2)/(2*nu^2));

% 2. Initialize solution
u = zeros(N, N);

% 3. Create FD Laplacian (sparse)
e = ones(N-2, 1);
D2 = spdiags([e -2*e e], -1:1, N-2, N-2)/dx^2;
I = speye(N-2);
A = kron(I, D2) + kron(D2, I);  % Kronecker product for 2D

% 4. Prepare RHS vector (interior points only)
f_inner = f(2:end-1, 2:end-1);
rhs = f_inner(:);

% 5. Solve system
u_inner = A \ rhs;

% 6. Insert solution
u(2:end-1, 2:end-1) = reshape(u_inner, [N-2, N-2]);

% % Visualization
% figure;
% surf(X, Y, u);
% title('FDM Solution');
% xlabel('x'); ylabel('y');
% shading interp; colorbar;
end
%------------------------------------------------------>
%------------------------------------------------------>
%------------------------------------------------------>

function plotComparisonWithErrorContour(u_pielm, u_exact, nu)
    x = linspace(0, 1, 100);
    y = linspace(0, 1, 100);
    [X, Y] = meshgrid(x, y);
    abs_err = abs(u_pielm - u_exact);
    squared_err = abs_err.^2;            
    mse = mean(squared_err, 'all');

    figure('Color', 'w', 'Position', [100 100 1800 600]); % Wide layout

    fontSize = 20;
    lineWidth = 1.5;

    % --- Subplot 1: PIELM Solution Surface ---
    subplot(1,3,1)
    surf(X, Y, u_pielm, 'EdgeColor', 'none');
    colormap(parula);
    colorbar('FontSize', fontSize, 'TickLabelInterpreter', 'latex');
    title(['KAPI-ELM, $\nu=' num2str(nu) '$, MSE=$' sprintf('%.2e', mse) '$'], ...
          'Interpreter', 'latex', 'FontSize', fontSize);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSize);
    zlabel('$u(x,y)$', 'Interpreter', 'latex', 'FontSize', fontSize);
    set(gca, 'FontSize', fontSize, 'TickLabelInterpreter', 'latex', ...
        'LineWidth', lineWidth, 'Box', 'on');
    view(3); % 3D view

    % --- Subplot 2: Exact Solution Surface ---
    subplot(1,3,2)
    surf(X, Y, u_exact, 'EdgeColor', 'none');
    colormap(parula);
    colorbar('FontSize', fontSize, 'TickLabelInterpreter', 'latex');
    title(['Finite Difference Method, $\nu=' num2str(nu) '$'], ...
          'Interpreter', 'latex', 'FontSize', fontSize);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSize);
    zlabel('$u(x,y)$', 'Interpreter', 'latex', 'FontSize', fontSize);
    set(gca, 'FontSize', fontSize, 'TickLabelInterpreter', 'latex', ...
        'LineWidth', lineWidth, 'Box', 'on');
    view(3);

    % --- Subplot 3: Absolute Error Surface ---
    subplot(1,3,3)
    surf(X, Y, abs_err, 'EdgeColor', 'none');
    colormap(parula);
    colorbar('FontSize', fontSize, 'TickLabelInterpreter', 'latex');
    title('Absolute Error', 'Interpreter', 'latex', 'FontSize', fontSize);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSize);
    zlabel('$|u_{pred} - u_{\mathrm{exact}}|$', 'Interpreter', 'latex', 'FontSize', fontSize);
    set(gca, 'FontSize', fontSize, 'TickLabelInterpreter', 'latex', ...
        'LineWidth', lineWidth, 'Box', 'on');
    view(3);

    filename = sprintf('comparison_surface3D_nu_%.3f.png', nu);
    exportgraphics(gcf, filename, 'Resolution', 300); % Save high-res PNG

end
