function [V, u] = br(M,opts,tol)
	% Return the optimal value function and policy for a given population (distribution M)
	% tol is the tolerance for the convergence of value iteration
	if nargin < 3 tol = 1e-1; end
	% opts is a option struct with the following fields
	% .r: reward function
	% .type: 'optimize' or 'evaluate'
	if ~isfield(opts, 'type') opts.type = 'optimize'; end
	type = opts.type;
	% .policy: policy to evaluate
	if strcmp(type, 'evaluate') 
		if ~isfield(opts, 'policy')
			error('Policy not provided');
		end
		u = opts.policy;
	end

	S = size(M, 1);
	A = reshape(1:S,S,1);
	% V = -1e5 * rand(size(M)); % initial value function
	V = -1e2 + abs(ones(size(M))); % initial value function

	iter = 1;
	while iter <= 1e3
		iter = iter + 1;
		V_old = V;
		for i =1:S
			i1 = mod(i,S) + 1;
			if strcmp(type, 'optimize')
				V(i,:,:) = max(opts.r(i,A,M) + (1-1/S) * (bsxfun(@times, A/S, V_old(i1,:,:)) + bsxfun(@times, (1-A/S), V_old(i,:,:))));
			elseif strcmp(type, 'evaluate')
				a = u(i,:,:);
				V(i,:,:) = opts.r(i,a,M) + (1-1/S) * (a/S .* V_old(i1,:,:) + (1-a/S) .* V_old(i,:,:));
			end
		end
		if norm(V(:) - V_old(:), Inf) < tol
			break;
		end
	end
	% disp(norm(V(:) - V_old(:), Inf));
	% disp(iter);

	% Get the optimal policy
	if strcmp(type, 'optimize')
		u = zeros(size(M));
		for i = 1:S
			i1 = mod(i,S) + 1;
			[~,u1] = max(opts.r(i,A,M) + (1-1/S) * (bsxfun(@times, A/S, V(i1,:,:)) + bsxfun(@times, (1-A/S), V(i,:,:))));
			u(i,:,:) = u1;
		end
	end
end
