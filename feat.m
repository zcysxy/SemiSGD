function psi = feat(dim)
	% sep = linspace(0, 2*pi, dim+1)';
	% psi = @(s) abs(cos(s*2*pi - sep(1:end-1))) / 2 * pi;
	sep = linspace(0, 1, dim+1)';
    fun = @(s) normpdf(tan((s)*pi),0,10);
    m = integral(fun, 0, 1);
	psi = @(s) fun(s-sep(1:end-1)) / m;
end
