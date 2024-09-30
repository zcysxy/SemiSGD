clc; close all;
%% Setup
% Defaults for axes
set(0, 'DefaultAxesFontSize', 15, 'DefaultAxesFontName', 'times', 'DefaultAxesFontWeight', 'bold', 'DefaultAxesLineWidth', 1.5)
% Defaults for plots
set(0, 'DefaultLineLineWidth', 4, 'DefaultAxesLineStyleOrder', '.-', 'DefaultLineMarkerSize', 20)
set(0, 'DefaultLineMarker', 'none')
% Defaults for text
set(0, 'DefaultTextInterpreter', 'latex', 'DefaultTextFontName', 'times', 'DefaultTextFontWeight', 'bold')
% Defaults for legend
set(0, 'DefaultLegendInterpreter', 'latex')

% Problem parameters
opts.S = 200;     % state space
S = opts.S;
opts.A = S;      % action space
A = opts.A;
opts.del = 1/S;  % state's gap
del = opts.del;
addpath('data')
if ~exist('tras', 'var')
    load('tras.mat')
end
M0 = cos(linspace(0,1,S))';
M0 = M0 ./ sum(M0); % initial M
% opts.Q0 = zeros(S,A);  % initial Q
% opts.V0 = zeros(S,1);  % initial V
% opts.s0 = 1;

% Training parameters
opts.epochs = 10;

% Load reference solution
if ~exist('m_opt', 'var')
	if ~exist('opt_model.mat', 'file')
		opt
	else
		load('opt_model.mat')
	end
	m_opt = reshape(m_opt, [S,1]);
end
opts.m_opt = m_opt;

% Helper functions
draw = @(p) find(cumsum(p) > rand(1), 1);
opts.GLIE = false;
opts.softmax = @(q, h) draw(exp((q-max(q))*h) / sum(exp((q-max(q))*h)));
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function %WARNING: del-dependent
opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*200/3))).^2 / 200; % reward function %WARNING: del-dependent
opts.method = 'det'; % 'sto'chastic or 'det'erministic
opts.P_sto = @(s,a) mod(s + (a/S > rand()) - 1, S) + 1; %WARNING: not used
opts.P_det = @(s_con,a) s_con + a * del;
err = @(M,m_opt) squeeze(sum((circshift(M,0)-m_opt).^2, 1));
opts.tol_ip = 1e-1; opts.tol_br = 1e-1;
% ip = @(v) normalize(1. / v, 1, 'norm', 1);
% br = @(M) (opts.bonus(1:S)' + 0.5*(1-M*S/3)) / del + 1;
% expl = @(u) squeeze(sum(((squeeze(u) - br(ip(squeeze(u)),opts)) * del).^2,1));

%% SemiSGD
opts.method = 'det';
opts.temp = 1e9;
% opts.GLIE = true;
% opts.temp = 1e9; % Decrease the temp to make SGD surpass OMD
opts.GLIE = false;
opts.alpha0 = 1e-3;
opts.beta0 = 1e-3;
% opts.T = 1.2e5;
opts.T = 1e4;
opts.K = 1e2; % the key is to keep T >= 1e3
opts.Q0 = Inf; opts.M0 = Inf, opts.s0 = Inf;
opts = rmfield(opts, {'Q0', 'M0', 's0'});
opts.S = 200;
opts.A = 200;
del = 1/opts.S;
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*200/3))).^2 / 200; % reward function
opts.P_det = @(s_con,a) s_con + a * del;
err = @(M,m_opt) squeeze(sum(abs(repelem(M, 200/opts.S, 1)-m_opt), 1));
fprintf('Running SemiGD\n')
% [M_gd_arr, Q_gd_arr] = gd(opts);
% err_gd = err(M_gd_arr, m_opt);
% [V_gd_arr, u_gd_arr] = max(Q_gd_arr, [], 2);
% err_V_gd = err(V_gd_arr, V_opt);
% expl_gd = expl(squeeze(u_gd_arr),opts);
% NOTE: log: 
% [1,true]: no oscillation, FPI=GD, OMD=FP
% [9,false] = [4,true]: FPI & OMD oscilate, GD drastically diverge from opt, FP slowly

%% SemiSGD w/ coarser grid
dim = 20;
opts.Q0 = Inf; opts.M0 = Inf, opts.s0 = Inf;
opts = rmfield(opts, {'Q0', 'M0', 's0'});
opts.S = dim;
opts.A = dim;
del = 1/opts.S;
opts.del = 1/200;
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*opts.S/3))).^2 / 200; % reward function
opts.P_det = @(s_con,a) s_con + a * del;
err = @(M,m_opt) squeeze(sum(abs(M-m_opt), 1));
fprintf('Running SemiGD with 10 states\n')
[M_cor, Q_cor] = gd(opts);
M_cor = repelem(M_cor, 200/opts.S, 1) * opts.S / 200;
err_cor = err(M_cor, m_opt);
% [V_gd_arr, u_gd_arr] = max(Q_gd_arr, [], 2);
% err_V_gd = err(V_gd_arr, V_opt);
% expl_gd = expl(squeeze(u_gd_arr),opts);

%% SemiSGD w/ LFA
opts.Q0 = Inf; opts.M0 = Inf, opts.s0 = Inf;
opts = rmfield(opts, {'Q0', 'M0', 's0'});
opts.S = dim;
opts.A = dim;
opts.dim = dim;
del = 1/opts.S;
opts.del = 1/200;
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*opts.S/3))).^2 / 200; % reward function
opts.P_det = @(s_con,a) s_con + a * del;
fprintf('Running SemiGD with 10 states\n')
[e_lfa, Q_lfa,M_lfa] = gd_lfa(opts);
err_lfa = err(M_lfa, m_opt);

%% Plot MSE
skip = 1;
ci = 0.8;
figure('visible', 'off'); hold on;
axis = gca;
varplot(err_cor(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'grid')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
% varplot(err_gd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', '200');
% axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
varplot(err_lfa(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'LFA')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
axis.YScale = 'log';
axis.XLim = [0, 100];
legend('show', 'fontsize', 18)
xlabel("Iterations", 'fontsize', 18)
ylabel("MSE", 'fontsize', 18)
exportgraphics(gca, sprintf('fig/lfa/mse_%d.png', dim), 'Resolution', 900)
savefig(sprintf('fig/lfa/mse_%d.fig', dim))
% title('Mean squared error', 'fontsize', 25)

%% Plot exploitability
%{
figure
axis = gca;
% Plot a line of constant 1e-4 for reference
varplot(expl_fpi(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'FPI')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(expl_er(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'FPI + ER')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(expl_fp(1:skip:end,:), 'ci', ci,'marker', 'none', 'DisplayName', 'FPI + FP')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(expl_omd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'FPI + OMD')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(expl_gd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'SemiSGD');
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
% hold on
% plot(expl(u_opt,opts)*ones(1:skip,opts.K), 'marker', 'none', 'DisplayName', 'Optimal')
axis.YScale = 'log';
axis.XLim = [0, 200];
% axis.YLim = [1e-2, 1];
legend('show', 'fontsize', 18)
% title('Exploitability', 'fontsize', 25)
%}

%% Plot distribution
figure('visible', 'off'); hold on;
axis = gca;
% varplot(squeeze(M_gd_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'DisplayName', '200')
% axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
varplot(squeeze(M_cor(:,end,:)), 'ci', ci, 'marker', 'none', 'DisplayName', 'grid')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
varplot(squeeze(M_lfa(:,end,:)), 'ci', ci, 'marker', 'none', 'DisplayName', 'LFA')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
% Plot m_opt ussing dash and circlr marker
plot(m_opt, 'LineStyle', '--', 'marker', 'none', 'MarkerSize', 8, 'DisplayName', 'MFE')
% axis.XLim = [0, 200];
% axis.YLim = [1e-2, 1];
yt=arrayfun(@num2str,get(gca,'ytick')*S,'un',0)
xt=arrayfun(@num2str,get(gca,'xtick')/200,'un',0)
set(gca,'yticklabel',yt);
set(gca,'xticklabel',xt);
legend('show', 'fontsize', 18, 'location', 'northwest')
xlabel("Iterations", 'fontsize', 18)
ylabel("Density", 'fontsize', 18)
exportgraphics(gca, sprintf('fig/lfa/mu_%d.png', dim), 'Resolution', 900)
savefig(sprintf('fig/lfa/mu_%d.fig', dim))
% title('Learned distributions', 'fontsize', 25)

%% Plot population distribution
%{
figure
set(0, 'DefaultLineLineWidth', 2);
axis = gca;
plot(scale(m_opt),  'DisplayName', 'Optimal')
hold on
plot(scale(M_on),  'DisplayName', 'FPI')
hold on
plot(scale(M_fp),  'DisplayName', 'FP')
hold on
plot(scale(M_omd), 'DisplayName', 'OMD')
hold on
plot(scale(circshift(M_gd,0)), 'DisplayName', 'GD');
title('Learned population distribution')
legend('show')
%}

% plot_list = {'T'};
% save_flag = false;
% plot_results

% mean(err_cor(:,end,:))
% std(err_cor(:,end,:))
% mean(err_lfa(:,end,:))
% std(err_lfa(:,end,:))
