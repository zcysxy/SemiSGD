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

opts.radius = 5;
opts.r = @(s,a,M) - ((a*del).^2 + 0.5 * (1 - neighbor_center(M,s,opts)).^2) * del; % reward function

% Training parameters
opts.epochs = 10;

% Helper functions
scale = @(arr) (arr - min(arr)) ./ (max(arr) - min(arr));
draw = @(p) find(cumsum(p) > rand(1), 1);
opts.GLIE = false;
opts.softmax = @(q, h) draw(exp((q-max(q))*h) / sum(exp((q-max(q))*h)));
opts.method = 'det'; % 'sto'chastic or 'det'erministic
opts.P_sto = @(s,a) mod(s + (a/S > rand()) - 1, S) + 1;
opts.P_det = @(s_con,a) s_con + a * del;
err = @(M,m_opt) squeeze(sum((circshift(M,0)-m_opt).^2, 1));
opts.tol_ip = 1e-1; opts.tol_br = 1e-1;

temp_list = [1e-4, 1e-2, 1e-0, 1e2, 1e4, 1e6, 1e8, 1e10];
temp_mask = [2, 3, 4, 5, 6, 8];

%% Load reference solution
if ~exist('m_opt', 'var')
	if ~exist('opt_model.mat', 'file')
		opt
	else
		load('opt_model.mat')
	end
	m_opt = reshape(m_opt, [S,1]);
end
opts.m_opt = m_opt;

%% SemiSGD
opts.method = 'det';
% opts.temp = 1e6; % Decrease the temp to make SGD surpass OMD
opts.GLIE = false;
% step size
opts.alpha0 = 1e-3;
opts.beta0 = 1e-3;
% opts.T = 1.2e5;
opts.T = 1e5;
opts.K = 2e2; % the key is to keep T >= 1e3

output = {};

for i = 1:length(temp_list)
	opts.temp = temp_list(i);
	fprintf('Running SemiGD\n')
	[M_gd_arr, Q_gd_arr] = gd(opts);
	output{i}.M_gd_arr = M_gd_arr;
	output{i}.Q_gd_arr = Q_gd_arr;
	output{i}.err_gd = err(M_gd_arr, m_opt);
	[V_gd_arr, u_gd_arr] = max(Q_gd_arr, [], 2);
	err_V_gd = err(V_gd_arr, V_opt);
	output{i}.expl_gd = expl(squeeze(u_gd_arr),opts);
end

% Save the data
save('data/semiSGD.mat', 'output', '-mat')

%% Plot MSE
skip = 1;
ci = 0.8;
figure
axis = gca;
for i = 1:length(temp_mask)
	varplot(output{temp_mask(i)}.err_gd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('%.0e', temp_list(temp_mask(i))))
	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
	hold on
end
axis.YScale = 'log';
axis.XLim = [0, 200];
% axis.YLim = [5e-5, 2e-2];
leg = legend('show', 'fontsize', 18);
title(leg, '$L_\pi$')
xlabel('Samples', 'FontSize', 18); ylabel('MSE', 'FontSize', 18)
xsecondarylabel('$\times 5\times 10^2$')
axisx = get(gca,'XAxis');
axisx.TickLabelInterpreter = 'latex';

%% Plot exploitability
figure
axis = gca;
for i = 1:length(temp_mask)
	varplot(output{temp_mask(i)}.expl_gd(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('%.0e', temp_list(temp_mask(i))))
	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
	hold on
end
axis.YScale = 'log';
axis.XLim = [0, 200];
% axis.YLim = [5e-5, 2e-2];
leg = legend('show', 'fontsize', 18);
title(leg, '$L_\pi$')
xlabel('Samples', 'FontSize', 18); ylabel('Exploitability', 'FontSize', 18)
xsecondarylabel('$\times 5\times 10^2$')
axisx = get(gca,'XAxis');
axisx.TickLabelInterpreter = 'latex';

%% Plot distribution
figure
axis = gca;
for i = 1:length(temp_mask)
	varplot(squeeze(output{temp_mask(i)}.M_gd_arr(:,end,:)), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('T=%g', temp_list(temp_mask(i))), 'HandleVisibility', 'off')
	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
	hold on
end
plot(m_opt, 'LineStyle', '--', 'marker', 'none', 'MarkerSize', 8, 'DisplayName', 'MFE')
axis.YScale = 'log';
% axis.XLim = [0, 200];
% axis.YLim = [5e-5, 2e-2];
legend('show', 'fontsize', 18)
xlabel('State space', 'FontSize', 18); ylabel('Population density', 'FontSize', 18)

