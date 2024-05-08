function [err,M_avg,Q_avg,v_avg] = gd(opts)

if ~isfield(opts, 'S') error('Missing S!'); end
if ~isfield(opts, 'A') error('Missing A!'); end
S = opts.S; A = opts.A;
if ~isfield(opts, 'del') opts.del = 1/S; end
if ~isfield(opts, 'Q0') opts.Q0 = zeros(S,1); end
if ~isfield(opts, 's0') opts.s0 = 1; end
if ~isfield(opts, 'm_opt') error('Missing m_opt!'); end
if ~isfield(opts, 'r') error('Missing r!'); end
if ~isfield(opts, 'GLIE') opts.GLIE = false; end
if ~isfield(opts, 'softmax') error('Missing softmax!'); end
if ~isfield(opts, 'epochs') opts.epochs = 10; end
if ~isfield(opts, 'T') opts.T = 1000; end
if ~isfield(opts, 'M0') opts.M0 = ones(opts.S,1) / opts.S; end
% if ~isfield(opts, 'FP') opts.FP = false; end
if ~isfield(opts, 'method') opts.method = 'det'; end
if ~isfield(opts, 'kappa') opts.kappa = 1; end
if ~isfield(opts, 'alpha0') opts.alpha0 = 1e-2; end
if ~isfield(opts, 'beta0') opts.beta0 = 1e-2; end
if ~isfield(opts, 'temp') opts.temp = 1e-3; end

del = opts.del;
M0 = opts.M0; Q0 = opts.Q0; s0 = opts.s0;
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

M_avg = zeros(S,1);
Q_avg = zeros(S,1);
v_avg = zeros(S,1);
err = zeros(T+1,epochs);

for e = 1:epochs
    fprintf('epoch: %d\n', e)
    Q = Q0;
    M = abs(randn(S,1));
    M = M ./ sum(M);
		M_hold = M;
    s1 = randi(S) - 1;              % random initial state
    s_con = s1;
    err(1,e) = sum(abs((circshift(M,-2)-m_opt)))^2;
    
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
		% Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r(s,a,M) + gamma * max(Q(s1,filter(s1,:) == 1)));
		if mod(t,1) == 0
			M_hold = M;
		end
		Q(s+1,a) = (1-alpha) * Q(s+1,a) + alpha * (r(s,a,M_hold) + (1-del) * Q(s1+1,a1));
		% Update M
		beta = beta0;%/ t;
		M = (1-beta) * M; M(s1+1) = M(s1+1) + beta * 1;
		err(t+1,e) = sum(abs((circshift(M,-2)-m_opt)))^2;
	end
	
	% Log error
	% err(t) = err(t) + sum(abs(M-m_opt)) * del;

	% % Exploitability
	% VS = Q';
	% VS = softmax(VS, 1e-4);
	% M_br = VS^1e3 * M;
	% V = max(Q,[],2);
	% V_br = V;
	% for t = 1:10
	% 	VS = repmat(V_br,1,A); % AxS
	% 	VS(filter' ~= 1) = -Inf;
	% 	V_br = r(1:S,1,M_br) + gamma * max(VS);
	% 	V_br = V_br';
	% end
	% expl(t,e) = sum(square((V_br-V)));
	M_avg = M_avg + M;
	Q_avg = Q_avg + Q;
	[~, v] = max(Q,[],2); % get deterministic policy from Q
	v_avg = v_avg + (v - 1);
end

% expl_qmi = expl;
% err = [sum(square((M0-m_opt)*S))/S; err];
M_avg = M_avg / epochs;
M_avg = M_avg(:,end);
M_avg = circshift(M_avg, -2);
Q_avg = Q_avg / epochs;
v_avg = round(v_avg / epochs);
[~, v_Q] = max(Q_avg,[],2);
% u_Q = u_Q - 1;
% u_opt = u_avg;
% m_opt = M_avg;
% save('opt.mat', 'M_avg', 'Q_avg', 'v_avg', '-mat')
end
