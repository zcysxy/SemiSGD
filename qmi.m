function [M_arr,Q_arr] = qmi(opts)
if ~isfield(opts, 'S') error('Missing S!'); end
if ~isfield(opts, 'A') error('Missing A!'); end
S = opts.S; A = opts.A;
if ~isfield(opts, 'del') opts.del = 1/S; end
if ~isfield(opts, 'policy') error('Missing policy!'); end
policy = opts.policy;
% if ~isfield(opts, 'Q0') opts.Q0 = zeros(S,1); end
% if ~isfield(opts, 's0') opts.s0 = 0; end
if ~isfield(opts, 'm_opt') error('Missing m_opt!'); end
if ~isfield(opts, 'r') error('Missing r!'); end
if ~isfield(opts, 'softmax') error('Missing softmax!'); end
if ~isfield(opts, 'temp') opts.temp = 1; end
if ~isfield(opts, 'GLIE') opts.GLIE = false; end
if ~isfield(opts, 'P_sto') error('Missing P_sto!'); end
if ~isfield(opts, 'P_det') error('Missing P_det!'); end
if ~isfield(opts, 'method') opts.method = 'det'; end
if ~isfield(opts, 'epochs') opts.epochs = 10; end
if ~isfield(opts, 'K') opts.K = 48; end
if ~isfield(opts, 'T') opts.T = 20; end
% if ~isfield(opts, 'M0') opts.M0 = ones(opts.S,1) / opts.S; end
if ~isfield(opts, 'kappa') opts.kappa = 2 + strcmpi(policy, 'on'); end
if ~isfield(opts, 'FP') opts.FP = false; end
if ~isfield(opts, 'OMD') opts.OMD = false; end

del = opts.del;
% M0 = opts.M0; Q0 = opts.Q0; s0 = opts.s0;
epochs = opts.epochs;
m_opt = opts.m_opt; r = opts.r;
softmax = opts.softmax; bonus = opts.bonus;
temp = opts.temp; GLIE = opts.GLIE;
P_sto = opts.P_sto; P_det = opts.P_det;
method = opts.method;
FP = opts.FP; OMD = opts.OMD;

kappa = opts.kappa;
skip = 1; % + strcmpi(policy, 'off');
K = opts.K * skip; 
T = opts.T;
% T = opts.T + (kappa * S - 1) * opts.T;
K0 = 200;
skip = K/K0;

M_arr = zeros(S,K0,epochs);
Q_arr = zeros(S,S,K0,epochs);

for e = 1:epochs
    fprintf("epoch: %d\n", e)
		if ~isfield(opts, 'Q0') Q = rand(S,A) * 1e-6; else Q = opts.Q0; end
		if ~isfield(opts, 'M0') M = abs(randn(S,1)); M = M./sum(M); else M = opts.M0; end
		if ~isfield(opts, 's0') s1 = randi(S); else s1 = opts.s0; end
    s_con = s1;
		% Outer iteration initialization
		Qk0 = Q; Mk0 = M;

		M_arr(:,1,e) = M;
		Q_arr(:,:,1,e) = Q;
    for k = 1:K
        threshold = (k > 15 * skip);
        for t = 1:T
						if GLIE temp_mult = t + k*T; else temp_mult = 1; end
            % Sample
            s = s1;
            if strcmpi(policy, 'on')
                a = softmax(Q(s,:),temp * temp_mult);
            elseif strcmpi(policy, 'off')
                a = softmax(Qk0(s,:),temp * temp_mult);
            end
            if strcmpi(method, 'sto')
                s1 = P_sto(s,a);
            elseif strcmpi(method, 'det')
                s_con = P_det(s_con,a);
                s1 = mod(round(s_con) - 1, S) + 1;
            end
            
            % Update Q
            % alpha = 1/(w*(t+c0));
            % alpha = 1e-2/(t*k * threshold + 1);
            % beta = 1/t/(k * (1 + 50 * threshold));
            % alpha = 0.1 / ((t) * threshold + 1);
            alpha = 1e-3;
            % beta = 1/t; %!! WARNING: let step sizes be consistent
            beta = 1e-3; %!! WARNING: let step sizes be consistent
						% beta = 1e-3;
            Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r(s,a,Mk0) + (1-del) * max(Q(s1,:))); % Greedy
            % Update M
            M = (1-beta) * M; M(s1) = M(s1) + beta * 1;
        end

        if FP
            M = (1-1/k)*Mk0 + 1/k * M;
						Mk0 = M;
        else
            Mk0 = M;
        end

				if OMD
					Q = (1-1/k)*Qk0 + 1/k * Q;
					Qk0 = Q;
				else
					Qk0 = Q;
				end
        
        % Log
				if mod(k, skip) == 0
					M_arr(:,k/skip+1,e) = M;
					Q_arr(:,:,k/skip+1,e) = Q;
				end
    end
end
end
