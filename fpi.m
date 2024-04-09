function [err,expl,M_avg,V_avg,v_avg] = fpi(opts)
    if ~isfield(opts, 'S'), error("Missing S!"); end
    if ~isfield(opts, 'A'), error("Missing A!"); end
    if ~isfield(opts, 'm_opt'), error("Missing m_opt!"); end
    if ~isfield(opts, 'r'), error("Missing r!"); end
    if ~isfield(opts, 'softmax'), error("Missing softmax!"); end
    if ~isfield(opts, 'bonus'), error("Missing bonus!"); end
    if ~isfield(opts, 'del'), opts.del = 1/opts.S; end
    if ~isfield(opts, 'T'), opts.T = 1e4; end
    if ~isfield(opts, 'K'), opts.K = 50; end
    if ~isfield(opts, 'V0'), opts.V0 = zeros(opts.S,1); end
    if ~isfield(opts, 'epochs'), opts.epochs = 10; end

    T = opts.T;
    K = opts.K;
    S = opts.S;
    A = opts.A;
    m_opt = opts.m_opt;
    r = opts.r;
    softmax = opts.softmax;
    bonus = opts.bonus;
    del = opts.del;
    V = opts.V0;
    epochs = opts.epochs;
    M_avg = zeros(S,1);
    V_avg = zeros(S,1);
    err = zeros(K+1,epochs);
    expl = zeros(K,epochs);

    for e = 1:epochs
        if ~isfield(opts, 'M0'), M = abs(randn(S,1)); M = M ./ sum(M); end
        fprintf("epoch: %d\n", e)
        err(1,e) = sum(abs((M-m_opt)))^2;
        for k = 1:K
            % Update V
            for t = 1:T
                for s = 0:S-1
                    V(s+1) = max(r(s,1:A,M) + (1-del) * ((1-[1:A]*del)*V(s+1) + (1:A)*del*V(mod(s+1,S)+1)));
                end
            end

            % Update M
            Mk0 = M;
            for t = 1:T
                Mt0 = M;
                % Behavior actions
                as = arrayfun(@(s) softmax(r(s,1:A,Mk0) + (1-del) * (1-[1:A]*del)*V(s+1) + (1:A)*del*V(mod(s+1,S)+1),1e2*k), 0:S-1);
                for s = 0:S-1
                    sl = mod(s-1,S); sr = mod(s+1,S);
                    M(s+1) = 1/2 * (Mt0(sl+1) * (1+as(sl+1)*del) + Mt0(sr+1) * (1-as(sr+1)*del));
                end
            end
            
            % Log error
            err(k+1,e) = sum(abs((M-m_opt)))^2 ;
            u_br = arrayfun(@(s) (bonus(s) + 0.5*(1-M(s+1)*S/3)) / del + 1, 0:S-1);
            u_cr = arrayfun(@(s) softmax(r(s,1:A,Mk0) + (1-del) * (1-[1:A]*del)*V(s+1) + (1:A)*del*V(mod(s+1,S)+1),1e6), 0:S-1);
            expl(k,e) = sum((abs(u_br - u_cr) * del).^2);
            % err(k) = err(k) + sum(abs(M-m_opt)) * del;
        end
        M_avg = M_avg + M;
        V_avg = V_avg + V;
    end

    M_avg = M_avg / epochs;
    V_avg = V_avg / epochs;
    v_avg = arrayfun(@(s) softmax(r(s,1:A,M_avg) + (1-del) * (1-[1:A]*del)*V_avg(sl+1) + (1:A)*del*V(mod(s+1,S)+1),1e2*k), 0:S-1)';
end
