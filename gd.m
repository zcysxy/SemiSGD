function [M_arr,Q_arr] = gd(opts)

if ~isfield(opts, 'S') error('Missing S!'); end
if ~isfield(opts, 'A') error('Missing A!'); end
S = opts.S; A = opts.A;
if ~isfield(opts, 'del') opts.del = 1/S; end
% if ~isfield(opts, 'M0') opts.M0 = ones(opts.S,1) / opts.S; end
% if ~isfield(opts, 'Q0') opts.Q0 = zeros(S,1); end
% if ~isfield(opts, 's0') opts.s0 = 1; end
if ~isfield(opts, 'm_opt') error('Missing m_opt!'); end
if ~isfield(opts, 'r') error('Missing r!'); end
if ~isfield(opts, 'GLIE') opts.GLIE = false; end
if ~isfield(opts, 'softmax') error('Missing softmax!'); end
if ~isfield(opts, 'epochs') opts.epochs = 10; end
if ~isfield(opts, 'T') opts.T = 1000; end
if ~isfield(opts, 'K') error('Missing K!'); end
% if ~isfield(opts, 'FP') opts.FP = false; end
if ~isfield(opts, 'method') opts.method = 'det'; end
if ~isfield(opts, 'kappa') opts.kappa = 1; end
if ~isfield(opts, 'alpha0') opts.alpha0 = 1e-2; end
if ~isfield(opts, 'beta0') opts.beta0 = 1e-2; end
if ~isfield(opts, 'temp') opts.temp = 1e-3; end

del = opts.del;
% M0 = opts.M0; Q0 = opts.Q0; s0 = opts.s0;
epochs = opts.epochs;
m_opt = opts.m_opt; r = opts.r;
softmax = opts.softmax;
P_sto = opts.P_sto; P_det = opts.P_det;
method = opts.method;
GLIE = opts.GLIE;
% FP = opts.FP;

% sample_comp = opts.kappa;
% T = opts.T + opts.T * (S * sample_comp-1);
T = opts.T; K = opts.K;
skip = T/K;
alpha0 = opts.alpha0; beta0 = opts.beta0;
temp = opts.temp;

M_arr = zeros(S,K+1,epochs);
Q_arr = zeros(S,S,K+1,epochs);

for e = 1:epochs
    fprintf('epoch: %d\n', e)
		if ~isfield(opts, 'Q0') Q = rand(S,A) * 1e-6; else Q = opts.Q0; end %WARNING: a random Q0 makes a huge difference
		if ~isfield(opts, 'M0') M = abs(randn(S,1)); M = M./sum(M); else M = opts.M0; end
		if ~isfield(opts, 's0') s1 = randi(S); else s1 = opts.s0; end
    s_con = s1;
		M_arr(:,1,e) = M;
		Q_arr(:,:,1,e) = Q;
    
	for t = 1:T
		if GLIE temp_mult = t; else temp_mult = 1; end
		% Sample
		s = s1;
		a = softmax(Q(s,:), temp * temp_mult);
		if strcmpi(method, 'sto')
				s1 = P_sto(s,a);
		elseif strcmpi(method, 'det')
				s_con = P_det(s_con,a);
				s1 = mod(round(s_con) - 1, S) + 1;
		end
		a1 = softmax(Q(s1,:), temp * temp_mult);
		
		% Update Q
		alpha = alpha0;%/ t;
		% Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r(s,a,M) + gamma * max(Q(s1,filter(s1,:) == 1)));
		Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r(s,a,M) + (1-del) * Q(s1,a1));
		% Update M
		beta = beta0;%/ t;
		M = (1-beta) * M; M(s1) = M(s1) + beta * 1;

		if mod(t, skip) == 0
			M_arr(:,t/skip+1,e) = M;
			Q_arr(:,:,t/skip+1,e) = Q;
		end
	end
end

end
