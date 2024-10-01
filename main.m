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
opts.S = 50;     % state space
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

%NOTE: log of rewards
% original, expl does not converge
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 1); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - opts.bonus(s) - 0.5*(1-M(s,:,:)*S/3)).^2 * del; % reward function
% expl converges, but all methods perform well
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - opts.bonus(s) - 0.5*(1-M(s,:,:)*S/3)).^2 * del; % reward function
% SGD performs the best for MSE, slightly worse than OMD in expl
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 1); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - max(opts.bonus(s), 0.5*(1-M(s,:,:)*S/3))).^2 * del; % reward function
% Better!
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*S/3))).^2 * del; % reward function
% opts.bonus = @(s) 0.1 * exp(s*del); % bonus function
% opts.pop_center = @(M) sum(M .* (1:opts.S)') * del;
opts.radius = 5;
opts.r = @(s,a,M) - ((a*del).^2 + 0.5 * (1 - neighbor_center(M,s,opts)).^2); % reward function
% opts.bonus = @(s)  (sin(4*pi*s*del) + 1); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - min(1, opts.bonus(s) * 0.5 *(1-M(s,:,:)*S/3))).^2 * del; % reward function

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
scale = @(arr) (arr - min(arr)) ./ (max(arr) - min(arr));
draw = @(p) find(cumsum(p) > rand(1), 1);
opts.GLIE = false;
opts.softmax = @(q, h) draw(exp((q-max(q))*h) / sum(exp((q-max(q))*h)));
opts.method = 'det'; % 'sto'chastic or 'det'erministic
opts.P_sto = @(s,a) mod(s + (a/S > rand()) - 1, S) + 1;
% opts.P_det = @(s_con,a) mod(s_con + (a-1) * del - 1, S) + 1;
opts.P_det = @(s_con,a) s_con + a * del;
err = @(M,m_opt) squeeze(sum((circshift(M,0)-m_opt).^2, 1));
opts.tol_ip = 1e-1; opts.tol_br = 1e-1;
% ip = @(v) normalize(1. / v, 1, 'norm', 1);
% br = @(M) (opts.bonus(1:S)' + 0.5*(1-M*S/3)) / del + 1;
% expl = @(u) squeeze(sum(((squeeze(u) - br(ip(squeeze(u)),opts)) * del).^2,1));

%% Run
%% LFA
%{
opts.dim = 50;
opts.method = 'det';
% opts.temp = 1e1;
% opts.temp = 1e-1;
% opts.GLIE = true;
opts.GLIE = false;
opts.alpha0 = 1e-3;
opts.beta0 = 1e-3;
opts.T = 2e4;
[err_gd, M_gd, Q_gd, V_gd] = gd_lfa(opts);
%}


%% SemiSGD
opts.method = 'det';
% opts.temp = 1e1;
% opts.GLIE = true;
% opts.temp = 1e9; % Decrease the temp to make SGD surpass OMD
opts.temp = 1e3; % Decrease the temp to make SGD surpass OMD
opts.GLIE = false;
opts.alpha0 = 1e-3;
opts.beta0 = 1e-3;
% opts.T = 1.2e5;
opts.T = 2e5;
opts.K = 2e3; % the key is to keep T >= 1e3
fprintf('Running SemiGD\n')
[M_gd_arr, Q_gd_arr] = gd(opts);
err_gd = err(M_gd_arr, m_opt);
[V_gd_arr, u_gd_arr] = max(Q_gd_arr, [], 2);
err_V_gd = err(V_gd_arr, V_opt);
expl_gd = expl(squeeze(u_gd_arr),opts);
% NOTE: log: 
% [1,true]: no oscillation, FPI=GD, OMD=FP
% [9,false] = [4,true]: FPI & OMD oscilate, GD drastically diverge from opt, FP slowly

%% SemiSGD w/ coarser grid
% opts.Q0 = Inf; opts.M0 = Inf, opts.s0 = Inf;
% opts = rmfield(opts, {'Q0', 'M0', 's0'});
% opts.S = 50;
% opts.A = 50;
% del = 1/opts.S;
% opts.del = 1/50;
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*S/3))).^2 * del; % reward function
% opts.P_det = @(s_con,a) s_con + a * del;
% err = @(M,m_opt) squeeze(sum((repelem(M, 50/opts.S, 1)-m_opt).^2, 1));
% fprintf('Running SemiGD with 10 states\n')
% [M_cor, Q_cor] = gd(opts);
% err_cor = err(M_cor, m_opt);
% [V_gd_arr, u_gd_arr] = max(Q_gd_arr, [], 2);
% err_V_gd = err(V_gd_arr, V_opt);
% expl_gd = expl(squeeze(u_gd_arr),opts);

%% Vanilla FPI
opts.TK = opts.T;
% opts.T = %1e3;%2e3;%400
opts.T = opts.TK / opts.K;
opts.policy = 'off';

fprintf('Running FPI\n')
opts.FP = false; opts.OMD = false;
[M_fpi_arr, Q_fpi_arr] = qmi(opts);
err_fpi = err(M_fpi_arr, m_opt);
[V_fpi_arr, u_fpi_arr] = max(Q_fpi_arr, [], 2);
err_V_fpi = err(V_fpi_arr, V_opt);
expl_fpi = expl(squeeze(u_fpi_arr),opts);

%% FPI + ER
fprintf('Running FPI w/ ER\n')
temp_hold = opts.temp;
opts.temp = temp_hold / 1e6;
[M_er, Q_er] = qmi(opts);
err_er = err(M_er, m_opt);
[V_er, u_er] = max(Q_er, [], 2);
err_V_er = err(V_er, V_opt);
expl_er = expl(squeeze(u_er),opts);
opts.temp = temp_hold;

%% FPI + FP
fprintf('Running FPI w/ FP\n')
opts.FP = true; opts.OMD = false;
[M_fp_arr, Q_fp_arr] = qmi(opts);
err_fp = err(M_fp_arr, m_opt);
[V_fp_arr, u_fp_arr] = max(Q_fp_arr, [], 2);
err_V_fp = err(V_fp_arr, V_opt);
expl_fp = expl(squeeze(u_fp_arr),opts);

%% FPI + OMD
fprintf('Running FPI w/ OMD\n')
opts.FP = false; opts.OMD = true;
[M_omd_arr, Q_omd_arr] = qmi(opts);
err_omd = err(M_omd_arr, m_opt);
[V_omd_arr, u_omd_arr] = max(Q_omd_arr, [], 2);
err_V_omd = err(V_omd_arr, V_opt);
expl_omd = expl(squeeze(u_omd_arr),opts);

%% Plot MSE
skip = 1;
ci = 0.8;
figure
axis = gca;
varplot(err_fpi(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'FPI')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_er(1:skip:end,:), 'ci', ci,'marker', 'none', 'DisplayName', 'FPI + ER')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_fp(1:skip:end,:), 'ci', ci,'marker', 'none', 'DisplayName', 'FPI + FP')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_omd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'FPI + OMD')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_gd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', 'SemiSGD');
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
axis.YScale = 'log';
% axis.XLim = [0, 200];
% axis.YLim = [5e-5, 2e-2];
legend('show', 'fontsize', 18)
% title('Mean squared error', 'fontsize', 25)

%% Plot exploitability
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

%% Plot distribution
figure
axis = gca;
varplot(squeeze(M_fpi_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'HandleVisibility', 'off')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(squeeze(M_er(:,end,:)), 'ci', ci, 'marker', 'none', 'HandleVisibility', 'off')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(squeeze(M_fp_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'HandleVisibility', 'off')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(squeeze(M_omd_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'HandleVisibility', 'off')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(squeeze(M_gd_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'HandleVisibility', 'off')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
% Plot m_opt ussing dash and circlr marker
plot(m_opt, 'LineStyle', '--', 'marker', 'none', 'MarkerSize', 8, 'DisplayName', 'MFE')
% axis.XLim = [0, 200];
% axis.YLim = [1e-2, 1];
yt=arrayfun(@num2str,get(gca,'ytick')*S,'un',0)
set(gca,'yticklabel',yt)
legend('show', 'fontsize', 18)
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
