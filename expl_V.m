function expl = expl_V(V,u,opts)
	[~,V_br] = br(ip(squeeze(u)),opts);
	expl = squeeze(sum(abs(V - V_br),1));
end
