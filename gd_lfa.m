function [err,M_avg,Q_avg,v_avg] = gd_lfa(opts)

if ~isfield(opts, 'S') error('etaissing S!'); end
if ~isfield(opts, 'A') error('etaissing A!'); end
if ~isfield(opts, 'dim') error('etaissing dimension!'); end
S = opts.S; A = opts.A; dim = opts.dim;
if ~isfield(opts, 'del') opts.del = 1/S; end
if ~isfield(opts, 'Q0') opts.Q0 = zeros(S,1); end
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
eta0 = opts.eta0; Q0 = opts.Q0; s0 = opts.s0;
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
alpha0 = opts.alpha0; beta0 = opts.beta0;
temp = opts.temp;

eta_avg = zeros(dim,1);
Q_avg = zeros(S,1);
v_avg = zeros(S,1);
err = zeros(T+1,epochs);

% Helper functions
psi = feat(dim);
psi_hat =  @(s) psi(s/S)/norm(psi((1:S)/S),1);
C_psi = integral(@(s) psi(s) *  psi_hat(s)', 0, 1, 'ArrayValued', true);
get_dist = @(eta) (psi((1:S)/S)' * eta) / sum(psi((1:S)/S)' * eta);

for e = 1:epochs
    fprintf('epoch: %d\n', e)
    Q = Q0;
    eta = eta0;
		eta_hold = eta;
    s1 = randi(S) - 1;              % random initial state
    s_con = s1;
    err(1,e) = sum(abs((circshift(get_dist(eta),-2)-m_opt)))^2;
    
	for t = 1:T
		if GLIE temp_mult = t; else temp_mult = 1; end
		% Sample
		s = s1;
		a = softmax(Q(s+1,:), temp * temp_mult);
		if strcmpi(method, 'sto')
				s1 = P_sto(s,a);
		elseif strcmpi(method, 'det')
				s_con = P_det(s_con,a);
				s1 = mod(round(s_con), S);
		end
		a1 = softmax(Q(s1+1,:), temp * temp_mult);
		
		% Update Q
		alpha = alpha0;%/ t;
		% Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r(s,a,eta) + gamma * max(Q(s1,filter(s1,:) == 1)));
		if mod(t,1) == 0
			eta_hold = eta;
		end
		Q(s+1,a) = (1-alpha) * Q(s+1,a) + alpha * (r(s,a,get_dist(eta_hold)) + (1-del) * Q(s1+1,a1));
		% Update eta
		beta = beta0;%/ t;
		% eta = projsplx(eta - beta * C_psi * eta + beta * psi_hat(s));
		eta = projsplx(eta - beta * C_psi * eta + beta * psi_hat(s));
		err(t+1,e) = sum(abs((circshift(get_dist(eta),-2)-m_opt)))^2;
	end
	
	% Log error
	% err(t) = err(t) + sum(abs(eta-m_opt)) * del;

	% % Exploitability
	% VS = Q';
	% VS = softmax(VS, 1e-4);
	% eta_br = VS^1e3 * eta;
	% V = max(Q,[],2);
	% V_br = V;
	% for t = 1:10
	% 	VS = repmat(V_br,1,A); % AxS
	% 	VS(filter' ~= 1) = -Inf;
	% 	V_br = r(1:S,1,eta_br) + gamma * max(VS);
	% 	V_br = V_br';
	% end
	% expl(t,e) = sum(square((V_br-V)));
	eta_avg = eta_avg + eta;
	Q_avg = Q_avg + Q;
	[~, v] = max(Q,[],2); % get deterministic policy from Q
	v_avg = v_avg + (v - 1);
end

% expl_qmi = expl;
% err = [sum(square((eta0-m_opt)*S))/S; err];
eta_avg = eta_avg / epochs;
eta_avg = eta_avg(:,end);
M_avg = circshift(get_dist(eta_avg), -2);
Q_avg = Q_avg / epochs;
v_avg = round(v_avg / epochs);
[~, v_Q] = max(Q_avg,[],2);
% u_Q = u_Q - 1;
% u_opt = u_avg;
% m_opt = eta_avg;
% save('opt.mat', 'eta_avg', 'Q_avg', 'v_avg', '-mat')
end
