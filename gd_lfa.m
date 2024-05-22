function [eta_arr,theta_arr,M_arr] = gd_lfa(opts)

if ~isfield(opts, 'S') error('etaissing S!'); end
if ~isfield(opts, 'A') error('etaissing A!'); end
if ~isfield(opts, 'dim') error('etaissing dimension!'); end
S = opts.S; A = opts.A; dim = opts.dim;
if ~isfield(opts, 'del') opts.del = 1/S; end
if ~isfield(opts, 's0') opts.s0 = 1; end
if ~isfield(opts, 'm_opt') error('etaissing m_opt!'); end
if ~isfield(opts, 'r') error('etaissing r!'); end
if ~isfield(opts, 'GLIE') opts.GLIE = false; end
if ~isfield(opts, 'softmax') error('etaissing softmax!'); end
if ~isfield(opts, 'epochs') opts.epochs = 10; end
if ~isfield(opts, 'T') opts.T = 1000; end
if ~isfield(opts, 'eta0') opts.eta0 = ones(opts.dim,1) / opts.dim; end
% if ~isfield(opts, 'FP') opts.FP = false; end
if ~isfield(opts, 'method') opts.method = 'det'; end
if ~isfield(opts, 'kappa') opts.kappa = 1; end
if ~isfield(opts, 'alpha0') opts.alpha0 = 1e-2; end
if ~isfield(opts, 'beta0') opts.beta0 = 1e-2; end
if ~isfield(opts, 'temp') opts.temp = 1e-3; end

del = opts.del;
eta0 = opts.eta0; s0 = opts.s0;
epochs = opts.epochs;
m_opt = opts.m_opt; r = opts.r;
softmax = opts.softmax;
P_sto = opts.P_sto; P_det = opts.P_det;
method = opts.method;
GLIE = opts.GLIE;
% FP = opts.FP;

% sample_comp = opts.kappa;
% T = opts.T + opts.T * (S * sample_comp-1);
T = opts.T;
K = opts.K;
skip = T/K;
alpha0 = opts.alpha0; beta0 = opts.beta0;
temp = opts.temp;

eta_arr = zeros(dim,K+1,epochs);
M_arr = zeros(50,K+1,epochs);
theta_arr = zeros(S,S,K+1,epochs);

% Helper functions
psi = feat(dim);
C_psi = integral(@(s) psi(s) *  psi(s)', 0, 1, 'ArrayValued', true);
get_dist_for_r = @(eta) (psi((1:S)/S)' * eta) / sum(psi((1:S)/S)' * eta);
get_dist = @(eta) (psi((1:50)/50)' * eta) / sum(psi((1:50)/50)' * eta);

for e = 1:epochs
    fprintf('epoch: %d\n', e)
		if ~isfield(opts, 'Q0') theta = rand(S,A) * 1e-6; else theta = opts.Q0; end %WARNING: a random Q0 makes a huge difference
		if ~isfield(opts, 'M0') eta = abs(randn(dim,1)); eta = eta./sum(eta); else eta = opts.M0; end
		if ~isfield(opts, 's0') s1 = randi(S); else s1 = opts.s0; end
		eta_hold = eta;
    s_con = s1;
		eta_arr(:,1,e) = eta;
		M_arr(:,1,e) = get_dist(eta);
		theta_arr(:,:,1,e) = theta;
    
	for t = 1:T
		if GLIE temp_mult = t; else temp_mult = 1; end
		% Sample
		s = s1;
		a = softmax(theta(s,:), temp * temp_mult);
		if strcmpi(method, 'sto')
				s1 = P_sto(s,a);
		elseif strcmpi(method, 'det')
				s_con = P_det(s_con,a);
				s1 = mod(round(s_con) - 1, S) + 1;
		end
		a1 = softmax(theta(s1,:), temp * temp_mult);
		
		% Update Q
		alpha = alpha0;%/ t;
		theta(s,a) = (1-alpha) * theta(s,a) + alpha * (r(s,a,get_dist_for_r(eta)) + (1-del) * theta(s1,a1));
		% Update eta
		beta = beta0;%/ t;
		% eta = projsplx(eta - beta * C_psi * eta + beta * psi_hat(s));
		eta = projsplx(eta - beta * C_psi * eta + beta * psi(s_con/S));

		if mod(t, skip) == 0
			eta_arr(:,t/skip+1,e) = eta;
			M_arr(:,t/skip+1,e) = get_dist(eta);
			theta_arr(:,:,t/skip+1,e) = theta;
		end
	end
end
end
