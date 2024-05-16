function M = ip(u,tol)
	% Return the induced population given a policy (speed u)
	% tol is the tolerance for the convergence 
	if nargin < 2; tol = 5e-2; end

	M = normalize(1./u, 1, 'norm', 1);
	S = size(u, 1);
	u = u / S;
	iter = 1;
	while iter <= 1e3
		iter = iter + 1;
		M_old = M;
		for i =1:S
			if i > 1; l = i - 1; else l = S; end
			if i < S; r = i + 1; else r = 1; end
			M(i,:,:) = 0.5 * (M_old(l,:,:) + M_old(r,:,:)) -...
							 0.5 * (M_old(r,:,:) .* u(r,:,:) -...
											M_old(l,:,:) .* u(l,:,:));
		end
		if norm(M(:) - M_old(:), Inf) < tol; break;
	end
	% disp(norm(M(:) - M_old(:), Inf))
end
