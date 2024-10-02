K_list = [2e2, 4e2, 8e2, 2e3, 1e4, 1e5];
output = {};

for i = 1:length(K_list)
	opts.K = K_list(i);
	main;
	% ouput{i}.M_gd = M_gd_arr;
	% ouput{i}.err_gd = err_gd;
	% ouput{i}.expl_gd = expl_gd;
	output{i}.M_fpi = M_fpi_arr;
	output{i}.err_fpi = err_fpi;
	output{i}.expl_fpi = expl_fpi;
end

%% Plot MSE
skip = 1;
ci = 0.85;
figure
axis = gca;
for i = 1:length(K_list)
	varplot(output{i}.err_fpi(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('%d', opts.TK / K_list(i)))
	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
	hold on
end
axis.YScale = 'log';
axis.XLim = [0, 200];
axis.YLim = [5e-5, 2e-2];
xlabel('Samples', 'fontsize', 18)
xsecondarylabel('$\times 5\times 10^2$')
ax = get(gca, 'XAxis');
ax.TickLabelInterpreter ='latex';
ylabel('MSE', 'fontsize', 18)
leg = legend('show', 'fontsize', 18);
title(leg, 'K')

%% Plot exploitability
skip = 2;
ci = 0.8;
figure
axis = gca;
for i = 1:length(K_list)
	varplot(output{i}.expl_fpi(1:skip:end,:), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('%d', opts.TK / K_list(i)))
	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
	hold on
end
axis.YScale = 'log';
axis.XLim = [0, 200/skip];
axis.YLim = [5e-5, 5e-2];
xlabel('Samples', 'fontsize', 18)
xsecondarylabel('$\times 10^3$')
ax = get(gca, 'XAxis');
ax.TickLabelInterpreter ='latex';
ylabel('Exploitability', 'fontsize', 18)
leg = legend('show', 'fontsize', 18);
title(leg, 'K')

% %% Plot distribution
% figure
% axis = gca;
% for i = 1:length(K_list)
% 	varplot(squeeze(output{i}.M_fpi(:,end,:)), 'ci', ci, 'marker', 'none', 'DisplayName', sprintf('%d', opts.TK / K_list(i)))
% 	axis.Children(1).EdgeColor = 'none'; axis.Children(1).FaceAlpha = 0.5; axis.Children(1).HandleVisibility = 'off';
% 	hold on
% end
% plot(m_opt, 'LineStyle', '--', 'marker', 'none', 'MarkerSize', 8, 'DisplayName', 'MFE')
% % axis.XLim = [0, 200];
% % axis.YLim = [1e-2, 1];
% yt=arrayfun(@num2str,get(gca,'ytick')*S,'un',0);
% set(gca,'yticklabel',yt)
% legend('show', 'fontsize', 18)
% xlabel('State space', 'fontsize', 18)
% ylabel('Population density', 'fontsize', 18)
