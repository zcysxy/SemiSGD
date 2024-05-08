clc; close all;

%% Setup
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
K = 50;
opts.K = 20;
opts.T = 20;
opts.TK = 1e5; %~= opts.T * opts.K * S * 10;
kappas = [1,2,3,4,5];
Ts = [50, 100, 125, 250, 5e2] * 2 * 1e1;

if ~exist('m_opt', 'var')
	if ~exist('opt.mat', 'file')
		opt
	end
	load('opt.mat')
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
%% SemiGD
opts.method = 'det';
% opts.temp = 1e1;
% opts.GLIE = false;
opts.temp = 1e-1;
opts.GLIE = true;
opts.alpha = 1e-3;
opts.beta0 = 1e-3;
opts.T = 1.2e5;
[err_gd, M_gd, Q_gd, V_gd] = gd(opts);

%% QMI w/o FP
opts.TK = opts.T;
opts.T = 2e3;%400;
opts.K = opts.TK / opts.T;
opts.policy = 'on';
opts.FP = false; opts.OMD = false; [err_on, expl_on, M_on, Q_on, ~] = qmi(opts);
opts.FP = true; opts.OMD = false;
[err_fp, expl_fp, M_fp, ~, ~] = qmi(opts);
opts.FP = false; opts.OMD = true;
[err_omd, expl_omd, M_omd, ~, ~] = qmi(opts);

%% Plot
figure
axis = gca;
varplot(err_on, 'marker', 'none', 'DisplayName', 'FPI')
hold on
varplot(err_fp, 'marker', 'none', 'DisplayName', 'FP')
hold on
varplot(err_omd, 'marker', 'none', 'DisplayName', 'OMD')
hold on
varplot(err_gd(1:floor(length(err_gd)/length(err_on)):end,:), 'DisplayName', 'GD');
axis.YScale = 'log';
legend('show')
title('Mean squared error')

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
plot(scale(M_gd), 'DisplayName', 'GD');
title('Learned population distribution')
legend('show')


% plot_list = {'T'};
% save_flag = false;
% plot_results
