function [u, V] = br(M,opts,tol)
	if nargin < 3 tol = 1e-1; end
	S = size(M, 1);
	A = reshape(1:S,S,1);
	V = -1e5 * ones(size(M));
	u = zeros(size(M));
	iter = 1;
	while iter <= 1e3
		iter = iter + 1;
		V_old = V;
		for i =1:S
			i1 = mod(i,S) + 1;
			V(i,:,:) = max(opts.r(i,A,M) + (1-1/S) * (bsxfun(@times, A/S, V_old(i1,:,:)) + bsxfun(@times, (1-A/S), V_old(i,:,:))));
		end
		if norm(V(:) - V_old(:), Inf) < tol
			break;
		end
	end
	% disp(norm(V(:) - V_old(:), Inf));
	for i = 1:S
		i1 = mod(i,S) + 1;
		[~,u1] = max(opts.r(i,A,M) + (1-1/S) * (bsxfun(@times, A/S, V(i1,:,:)) + bsxfun(@times, (1-A/S), V(i,:,:))));
		u(i,:,:) = u1;
	end
end
