function [] = pretty(ax,opts)
	ax.TickLength = [0, 0]; % remove the ticks
	ax.Legend.Box = 'off'; % remove the box around the legend
	if isfield(opts,'NumTicks')
		ax.XTick = linspace(ax.XLim(1),ax.XLim(2),opts.numTicks); % set the y-ticks
	end

	% Line
	if isfield(opts, 'AxisLineWidth'); t = opts.AxisLineWidth; else t = 3; end
	if isfield(opts, 'AxisLineWidth'); t = opts.AxisLineWidth; else t = 3; end
	
	% Font
	if isfield(opts, 'FontName'); t = opts.FontName; else t = 'Times New Roman'; end
	ax.FontName = t;

	if isfield(opts, 'FontWeight') t = opts.FontWeight; else t = 'Normal'; end
	ax.FontWeight = t;

	if isfield(opts, 'AxisFontSize') t = opts.AxisFontSize; else t = 20; end
  ax.FontSize = t;

	if isfield(opts, 'LegendFontSize') t = opts.AxisFontSize; else t = 20; end
	ax.Legend.FontSize = t;
end
