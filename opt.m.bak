clc; close all;

S=50; % States
iters=5e3; % Iterations
inner_iters=1e2; % Inner iterations
delta_t=1/S;
ini_rho = ones(S,1);
rho=zeros(S);
rho(:,1) = (ini_rho ./ sum(ini_rho));
u=zeros(S);
u_hist=zeros(iter,S);
rho_hist=zeros(iter,S);

for iter=1:iters
	% Update pop
	for inner_iter = 1:inner_iters
		rho_old = rho;
		for i =1:S
			if i > 1; l = i - 1; else l = S; end
			if i < S; r = i + 1; else r = 1; end
			rho(i)=0.5 * (rho_old(l) + rho_old(r)) -...
							 0.5 * (rho_old(r) * u(r) -...
											rho_old(l) * u(l));
		end
	end
	for i=1:S
		u(i) = 0.5*(1 - rho(i)*S/3) + 0.2 * (sin(4*pi*i/S) + 1); %!!WARNING: Closed form via the reward function
		u_hist(iter,i)=u(i);
		rho_hist(iter,i)=rho(i);
	end
	u=squeeze(sum(u_hist, 1))/iter; % OMD
	rho=squeeze(sum(rho_hist, 1))/iter; % FP
end

m_opt = rho;
u_opt = u;
save('opt.mat', 'u_opt', 'm_opt', '-mat')
% addpath('ring_road_04_04/')
% plot_3D
