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
    load('opt.mat')
end
opts.m_opt = m_opt;

% Helper functions
scale = @(arr) (arr - min(arr)) ./ (max(arr) - min(arr));
draw = @(p) find(cumsum(p) > rand(1), 1);
opts.softmax = @(q, h) draw(exp((q-max(q))*h) / sum(exp((q-max(q))*h)));
opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 1); % bonus function
opts.r = @(s,a,M) - 1/2*((a-1)*del - opts.bonus(s) - 0.5*(1-M(s+1)*S/3)).^2 * del; % reward function
opts.method = 'det'; % 'sto'chatic or 'det'erministic
opts.P_sto = @(s,a) mod(s + (a/S > rand()), S);
opts.P_det = @(s_con,a) mod(s_con + (a-1) * del, S);

%% Run
% QMI w/o FP
opts.policy = 'on';
opts.FP = false;

%% Off-policy QMI
opts.policy = 'off';
results = struct;
% for kappa = kappas
for T = Ts
    % opts.kappa = kappa;
    opts.T = T;
    opts.K = opts.TK / T;
    opts.kappa = 1/S;
    [err_off, expl_off, M_off, ~, ~] = qmi(opts);
    % results.(sprintf('kappa%d', kappa)) = struct('err', err_off, 'expl', expl_off, 'M', M_off);
    results.(sprintf('T%d', T)) = struct('err', err_off, 'expl', expl_off, 'M', M_off);
end

%% On-policy QMI
% opts.policy = 'on';
% results = struct;
% % for kappa = kappas
% for T = Ts
%     % opts.kappa = kappa;
%     opts.T = T;
%     opts.K = opts.TK / T;
%     opts.kappa = 1/S;
%     [err_on, expl_on, M_on, ~, ~] = qmi(opts);
%     % results.(sprintf('kappa%d', kappa)) = struct('err', err_on, 'expl', expl_on, 'M', M_on);
%     results.(sprintf('T%d', T)) = struct('err', err_on, 'expl', expl_on, 'M', M_on);
% end

%% FPI
% opts.K = K;
% opts.T = 20;
% [err_fpi, expl_fpi, M_fpi, V_avg, v_avg] = fpi(opts);
% err_fpi_line = repmat(err_fpi(end,:), [size(err_fpi,1),1]);
% expl_fpi_line = repmat(expl_fpi(end,:), [size(expl_fpi,1),1]);

%% Plot
plot_list = {'T'};
save_flag = false;
plot_results
