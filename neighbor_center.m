function center = neighbor_center(M,s,opts)
	neighbor = (s-opts.radius:s+opts.radius);
	neighbor = intersect(neighbor, 1:opts.S);
	neighbor_dist = M(neighbor,:,:);
	center = sum(neighbor_dist .* neighbor') ./ sum(neighbor_dist) * opts.del;
end
