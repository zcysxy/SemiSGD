clc; close all;
%% Setup
% Defaults for axes
set(0, 'DefaultAxesFontSize', 15, 'DefaultAxesFontName', 'times', 'DefaultAxesFontWeight', 'bold', 'DefaultAxesLineWidth', 1.5)
% Defaults for plots
set(0, 'DefaultLineLineWidth', 2, 'DefaultAxesLineStyleOrder', '.-', 'DefaultLineMarkerSize', 20)
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
if ~exist('tras', 'var')
    load('ring_road/tras.mat')
end
M0 = cos(linspace(0,1,S))';
M0 = M0 ./ sum(M0); % initial M
opts.Q0 = zeros(S,A);  % initial Q
opts.V0 = zeros(S,1);  % initial V
opts.s0 = 0;

% Training parameters
opts.epochs = 10;

% Load reference solution
if ~exist('m_opt', 'var')
	if ~exist('opt_fp_5e7.mat', 'file')
		opt
	end
	load('opt_fp_5e7.mat')
	m_opt = reshape(m_opt, [S,1]);
end
opts.m_opt = m_opt;

% Helper functions
scale = @(arr) (arr - min(arr)) ./ (max(arr) - min(arr));
draw = @(p) find(cumsum(p) > rand(1), 1);
opts.GLIE = false;
opts.softmax = @(q, h) draw(exp((q-max(q))*h) / sum(exp((q-max(q))*h)));
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 1); % bonus function
opts.r = @(s,a,M) - 1/2*((a-1)*del - opts.bonus(s) - 0.5*(1-M(s+1)*S/3)).^2 * del; % reward function
opts.method = 'det'; % 'sto'chastic or 'det'erministic
opts.P_sto = @(s,a) mod(s + (a/S > rand()), S);
opts.P_det = @(s_con,a) mod(s_con + (a-1) * del, S);

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

%% SemiGD
opts.method = 'det';
opts.temp = 1e1;
opts.GLIE = true;
% opts.temp = 1e9;
% opts.GLIE = false;
opts.alpha0 = 8e-4;
opts.beta0 = 8e-4;
% opts.T = 1.2e5;
opts.T = 1e6;
[err_gd, M_gd, Q_gd, V_gd] = gd(opts);
% NOTE: log: 
% [1,true]: no oscillation, FPI=GD, OMD=FP
% [9,false] = [4,true]: FPI & OMD oscilate, GD drastically diverge from opt, FP slowly

%% QMI w/o FP
opts.TK = opts.T;
opts.T = 1e3;%2e3;%400
opts.K = opts.TK / opts.T;
opts.policy = 'on';
opts.FP = false; opts.OMD = false;
[err_on, expl_on, M_on, Q_on, ~] = qmi(opts);
opts.FP = true; opts.OMD = false;
[err_fp, expl_fp, M_fp, ~, ~] = qmi(opts);
opts.FP = false; opts.OMD = true;
[err_omd, expl_omd, M_omd, ~, ~] = qmi(opts);

%% Plot
figure
skip = 2;
axis = gca;
varplot(err_on(1:skip:end,:), 'marker', 'none', 'DisplayName', 'FPI')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_fp(1:skip:end,:), 'marker', 'none', 'DisplayName', 'FPI + FP')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_omd(1:skip:end,:), 'marker', 'none', 'DisplayName', 'FPI + OMD')
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
hold on
varplot(err_gd(1:skip*floor(length(err_gd)/length(err_on)):end,:),  'marker', 'none', 'DisplayName', 'SemiSGD');
axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
% varplot(err_gd(1:floor(length(err_gd)/length(err_on)):end,:),  'marker', 'none', 'DisplayName', 'SemiSGD');
% axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.2; axis.Children(1).HandleVisibility = 'off';
axis.YScale = 'log';
axis.XLim = [0, 500];
legend('show', 'fontsize', 18)
title('Mean squared error', 'fontsize', 25)

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


% plot_list = {'T'};
% save_flag = false;
% plot_results
