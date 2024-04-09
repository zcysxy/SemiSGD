function [err,expl,M_avg,Q_avg,v_avg] = qmi(opts)
if ~isfield(opts, 'S') error('Missing S!'); end
if ~isfield(opts, 'A') error('Missing A!'); end
S = opts.S; A = opts.A;
if ~isfield(opts, 'del') opts.del = 1/S; end
if ~isfield(opts, 'policy') error('Missing policy!'); end
if ~isfield(opts, 'Q0') opts.Q0 = zeros(S,1); end
if ~isfield(opts, 's0') opts.s0 = 0; end
if ~isfield(opts, 'm_opt') error('Missing m_opt!'); end
if ~isfield(opts, 'r') error('Missing r!'); end
if ~isfield(opts, 'softmax') error('Missing softmax!'); end
if ~isfield(opts, 'P_sto') error('Missing P_sto!'); end
if ~isfield(opts, 'P_det') error('Missing P_det!'); end
if ~isfield(opts, 'method') opts.method = 'det'; end
if ~isfield(opts, 'epochs') opts.epochs = 10; end
if ~isfield(opts, 'K') opts.K = 48; end
if ~isfield(opts, 'T') opts.T = 20; end
if ~isfield(opts, 'M0') opts.M0 = ones(opts.S,1) / opts.S; end
if ~isfield(opts, 'kappa') opts.kappa = 2 + strcmpi(policy, 'on'); end


del = opts.del;
M0 = opts.M0; Q0 = opts.Q0; s0 = opts.s0;
epochs = opts.epochs;
m_opt = opts.m_opt; r = opts.r;
softmax = opts.softmax; bonus = opts.bonus;
P_sto = opts.P_sto; P_det = opts.P_det;
method = opts.method; policy = opts.policy;

kappa = opts.kappa;
skip = 1 + strcmpi(policy, 'off');
K = opts.K * skip; 
T = opts.T + (kappa * S - 1) * opts.T;


M_avg = zeros(S,K);
Q_avg = zeros(S,1);
v_avg = zeros(S,1);
if ~exist('m_opt', 'var')
    load('opt.mat')
end
err = zeros(K+1,epochs);
expl = zeros(K,epochs);

for e = 1:epochs
    fprintf("epoch: %d\n", e)
    Q = Q0;
    M = abs(randn(S,1));
    M = M ./ sum(M);
    s1 = randi(S) - 1;              % random initial state
    s_con = s1;

    err(1,e) = sum(abs((circshift(M,-1)-m_opt)))^2;
    for k = 1:K
        % Outer iteration initialization
        Qk0 = Q; Mk0 = M;
        threshold = (k > 15 * skip);
        for t = 1:T
            % Sample
            s = s1;
            if strcmpi(policy, 'on')
                a = softmax(Q(s+1,:),(t + k*T)/skip);
            elseif strcmpi(policy, 'off')
                a = softmax(Qk0(s+1,:),(t + k*T)/2);
            end
            if strcmpi(method, 'sto')
                s1 = P_sto(s,a);
            elseif strcmpi(method, 'det')
                s_con = P_det(s_con,a);
                s1 = mod(round(s_con), S);
            end
            
            % Update Q
            % alpha = 1/(w*(t+c0));
            % alpha = 1e-2/(t*k * threshold + 1);
            % beta = 1/t/(k * (1 + 50 * threshold));
            % alpha = 0.1 / ((t) * threshold + 1);
            alpha = 1e-3;
            beta = 1/t;
            Q(s+1,a) = (1-alpha) * Q(s+1,a) + alpha * (r(s,a,Mk0) + (1-del) * max(Q(s1+1,:))); % Greedy
            % Update M
            M = (1-beta) * M; M(s1+1) = M(s1+1) + beta * 1;
        end
        
        % Log error
        M_avg(:,k) = M_avg(:,k) + M;
        err(k+1,e) = sum(abs((circshift(M,-1)-m_opt)))^2;
        u_br = arrayfun(@(s) (bonus(s) + 0.5*(1-M(s+1)*S/3)) / del + 1, 0:S-1)';
        [~, u_cr] = max(Qk0,[],2);
        expl(k,e) = sum((abs(u_br - u_cr) * del).^2);
        % err(k) = err(k) + sum(abs(M-m_opt)) * del;
    end
    Q_avg = Q_avg + Q;
    [~, v] = max(Q,[],2); % get deterministic policy from Q
    v_avg = v_avg + (v - 1);
end

M_avg = M_avg / epochs;
% err = sum(abs(M_avg - m_opt)).^2;
M_avg = M_avg(:,end);
M_avg = circshift(M_avg, -2);
Q_avg = Q_avg / epochs;
v_avg = round(v_avg / epochs);
[~, v_Q] = max(Q_avg,[],2);
% u_Q = u_Q - 1;
% u_opt = u_avg;
% m_opt = M_avg;
% save('opt.mat', 'u_opt', 'm_opt', '-mat')
end
