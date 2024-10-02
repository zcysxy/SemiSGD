clc; close all;

S=50; % States
iters=30; % Iterations
del=1/S;
ini_rho = rand(S,1);
rho = ini_rho ./ sum(ini_rho);
% rho=zeros(S);
% rho(:,1) = (ini_rho ./ sum(ini_rho));
u_hist=zeros(S,iters);
V_hist=zeros(S,iters);
rho_hist=zeros(S,iters);
% bonus = @(s)  (sin(4*pi*s*del) + 1); % bonus function
% % bonus = rand(S,1) * 0.4;
% opts.r = @(s,a,M) - 1/2*(a*del - min(1, bonus(s) * 0.5 *(1-M(s,:,:)*S/3))).^2 * del; % reward function
% bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - bonus(s) - 0.5*(1-M(s,:,:)*S/3)).^2 * del; % reward function
% opts.bonus = @(s) 0.2 * (sin(4*pi*s*del) + 2); % bonus function
% opts.r = @(s,a,M) - 1/2*(a*del - min(opts.bonus(s), 0.5*(1-M(s,:,:)*S/3))).^2 * del; % reward function

for iter=1:iters
	if mod(iter, 10) == 0 disp(iter); end
	[V,u] = br(rho, opts,1e-1);
	u_hist(:,iter)=u;
	V_hist(:,iter)=V;
	% OMD
	u = squeeze(sum(u_hist, 2))/iter;
	V = squeeze(sum(V_hist, 2))/iter;

	rho = ip(u, 1e-1);
	rho_hist(:,iter)=rho;
	% FP
	rho = squeeze(sum(rho_hist, 2))/iter;
end

m_opt = rho;
u_opt = u;
V_opt = V;
figure; plot(expl(u_hist,opts)); xlabel('Iterations', 'FontSize', 18); ylabel('Exploitability', 'FontSize', 18);
% figure; plot(m_opt);
% save('opt.mat', 'u_opt', 'm_opt', '-mat')
% addpath('ring_road_04_04/')
% plot_3D
