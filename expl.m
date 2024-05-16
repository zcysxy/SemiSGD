function expl = expl(u,opts)
	% Compute the exploitability of a policy (speed u)
	if ~isfield(opts,'tol_br') opts.tol_br = 1e-1; end
	if ~isfield(opts,'tol_ip') opts.tol_ip = 1e-1; end

	% Calculate the induced policy of u
	mu_u = ip(squeeze(u),opts.tol_ip);

	% Calculate the value function of u given the fixed population mu_u
	opts.type = 'evaluate';
	opts.policy = u;
	V_u = br(mu_u,opts,opts.tol_br);

	% Calculate the optimal value function given the fixed population mu_u
	opts.type = 'optimize';
	V_br = br(mu_u,opts,opts.tol_br);

	% expl = squeeze(sum(abs(V_u - V_br),1));
	expl = squeeze(sum(mu_u .* (V_br - V_u),1));
end
