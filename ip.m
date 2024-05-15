function M = ip(u)
	M = normalize(1./u, 1, 'norm', 1);
	S = size(u, 1);
	u = u * 1/S;
	for k = 1:100
		M_old = M;
		for i =1:S
			if i > 1; l = i - 1; else l = S; end
			if i < S; r = i + 1; else r = 1; end
			M(i,:,:) = 0.5 * (M_old(l,:,:) + M_old(r,:,:)) -...
							 0.5 * (M_old(r,:,:) .* u(r,:,:) -...
											M_old(l,:,:) .* u(l,:,:));
		end
	end
end
