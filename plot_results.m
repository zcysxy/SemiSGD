%% Defaults
% Defaults for axes
set(0, 'DefaultAxesFontSize', 15, 'DefaultAxesFontName', 'times', 'DefaultAxesFontWeight', 'bold', 'DefaultAxesLineWidth', 1.5)
% Defaults for plots
set(0, 'DefaultLineLineWidth', 2, 'DefaultAxesLineStyleOrder', '.-', 'DefaultLineMarkerSize', 20)
set(0, 'DefaultLineMarker', 'none')
% Defaults for text
set(0, 'DefaultTextInterpreter', 'latex', 'DefaultTextFontName', 'times', 'DefaultTextFontWeight', 'bold')
% Defaults for legend
set(0, 'DefaultLegendInterpreter', 'latex')

xx = linspace(0,1-del,S);

%%
if ismember('policy', plot_list)
figure
% plot(xx, [v_avg*del, bonus(xx'/del) + 0.5*(1-M_avg/max(M_avg))], 'marker', 'none') % LTG
plot(xx, [v_avg*del, bonus(xx'/del) + 0.5*(1-M_avg*S/3)], 'marker', 'none') % STG
legend('learned policy', 'best response')
xlabel('state space', 'FontSize', 20)
ylabel('action', 'FontSize', 20)
title('Policy', 'FontSize', 25)
% exportgraphics(gca, 'u_5e2.png', 'Resolution', 900)
end

%%
if ismember('population', plot_list)
figure
ax = gca;
plot(xx, [scale((m_opt)), scale((M_fpi)), scale((M_off)), scale((M_on))], 'marker', 'none')
ax.ColorOrder = [255, 127, 0; 77, 175, 74; 55, 126, 184; 228, 26, 28]  / 255;
% plot(xx, [scale(smooth(M_avg)), scale(1./v_avg)], 'marker', 'none')
legend('equilibrium',  'FPI', 'off-policy QMI', 'on-policy QMI', 'Location', 'southeast')
xlabel('states', 'FontSize', 20)
ylabel('density', 'FontSize', 20)
% title('Population', 'FontSize', 25)
exportgraphics(gca, '../fig/rr/m.png', 'Resolution', 900)
savefig('../fig/rr/m.fig')
end

%%
if ismember('opt', plot_list)
figure
plot(xx, [M_avg, M0] ./ del, 'marker', 'none')
legend('induced population', 'initial population')
xlabel('state space', 'FontSize', 20)
ylabel('density', 'FontSize', 20)
title('Population', 'FontSize', 25)
% exportgraphics(gca, 'rho_ini.png', 'Resolution', 900)
end

%%
if ismember('error', plot_list)
%{
f = figure("Position", [100,100,1100,500]);
t = tiledlayout(1,2,"TileSpacing","compact");
nexttile
semilogy(2:K/2,err(2:K/2), 'Marker', 'none')
% hold on; semilogy(1:K/2,err_sto(1:K/2), 'Marker', 'none'); legend('deterministic', 'stochastic')
nexttile
semilogy(K/2:K,err(K/2:K), 'Marker', 'none')
% hold on; semilogy(K/2:K,err_sto(K/2:K), 'Marker', 'none')
%}
figure
axis = gca;
if exist('err_on', 'var')
    varplot(err_fpi, 'marker', 'none', 'color', '#4daf4a')
    hold on
    varplot(err_off(1:2:end,:), 'marker', 'none', 'color', '#377eb8')
    hold on
    varplot(err_on, 'marker', 'none', 'color', '#e41a1c')
    legend('FPI', '', 'off-policy QMI', '', 'on-policy QMI', '')
    axis.Children(3).EdgeColor = 'none';
    axis.Children(5).EdgeColor = 'none';
else
    varplot(err_off(1:2:end,:), 'marker', 'none', 'color', '#b2182b')
    legend('QMI', '')
end
axis.Children(1).EdgeColor = 'none';
axis.YScale = 'log';

xlabel('iterations', 'FontSize', 20)
ylabel('MSE', 'FontSize', 20)
% sgtitle('Error', 'FontSize', 25)
set(get(gca, 'XAxis'), 'Exponent', 0)
exportgraphics(gca, '../fig/rr/err.png', 'Resolution', 900)
savefig('../fig/rr/err.fig')
end


%% Exploitability
if ismember('expl', plot_list)
figure
axis = gca;
if exist('expl_on', 'var')
    varplot(expl_fpi, 'marker', 'none', 'color', '#4daf4a')
    hold on
    varplot(expl_off(1:2:end,:), 'marker', 'none', 'color', '#377eb8')
    hold on
    varplot(expl_on, 'marker', 'none', 'color', '#e41a1c')
    legend('FPI', '', 'off-policy QMI', '', 'on-policy QMI', '')
    axis.Children(3).EdgeColor = 'none';
    axis.Children(5).EdgeColor = 'none';
else
    varplot(expl_off(1:2:end,:), 'marker', 'none', 'color', '#b2182b')
    legend('QMI', '')
end
axis.Children(1).EdgeColor = 'none';
axis.YScale = 'log';

xlabel('iterations', 'FontSize', 20)
ylabel('exploitability', 'FontSize', 20)
% sgtitle('Expl', 'FontSize', 25)
set(get(gca, 'XAxis'), 'Exponent', 0)
exportgraphics(gca, '../fig/rr/expl.png', 'Resolution', 900)
savefig('../fig/rr/expl.fig')
end

%% Sample compensation factor
if ismember('comp', plot_list)
    f = figure();
    axis = gca;
    varplot(err_fpi_line, 'r--', 'displayname', 'FPI')
    axis.Children(1).EdgeColor = 'none';
    axis.Children(1).FaceAlpha = 0.2;
    axis.Children(1).HandleVisibility = 'off';
    hold on
    for kappa = kappas
        err = results.(sprintf('kappa%d', kappa)).err;
        if strcmp(opts.policy, 'off')
            err = err(1:2:end,1:min(opts.epochs,10));
        end
        varplot(err, 'marker', 'none', 'displayname', sprintf('%d', kappa))
        hold on
        axis.Children(1).EdgeColor = 'none';
        axis.Children(1).FaceAlpha = 0.4;
        axis.Children(1).HandleVisibility = 'off';
    end
    axis.YScale = 'log';
    axis.XLim = [1, opts.K];
    legend
    title(legend,'$\eta$');
    xlabel('iterations', 'FontSize', 20)
    ylabel('MSE', 'FontSize', 20)
    set(get(gca, 'XAxis'), 'Exponent', 0)
    exportgraphics(gca, sprintf('../fig/rr/comp_err_%s.png', opts.policy), 'Resolution', 900)
    savefig(sprintf('../fig/rr/comp_err_%s.fig', opts.policy))

    f = figure();
    axis = gca;
    varplot(expl_fpi_line, 'r--', 'displayname', 'FPI')
    axis.Children(1).EdgeColor = 'none';
    axis.Children(1).FaceAlpha = 0.2;
    axis.Children(1).HandleVisibility = 'off';
    hold on
    for kappa = kappas
        expl = results.(sprintf('kappa%d', kappa)).expl;
        if strcmp(opts.policy, 'off')
            expl = expl(1:2:end,:);
        end
        varplot(expl, 'marker', 'none', 'displayname', sprintf('%d', kappa))
        hold on
        axis.Children(1).EdgeColor = 'none';
        axis.Children(1).FaceAlpha = 0.4;
        axis.Children(1).HandleVisibility = 'off';
    end
    axis.YScale = 'log';
    axis.XLim = [1, opts.K];
    legend
    title(legend,'$\eta$');
    xlabel('iterations', 'FontSize', 20)
    ylabel('exploitability', 'FontSize', 20)
    set(get(gca, 'XAxis'), 'Exponent', 0)
    exportgraphics(gca, sprintf('../fig/rr/comp_expl_%s.png', opts.policy), 'Resolution', 900)
    savefig(sprintf('../fig/rr/comp_expl_%s.fig', opts.policy))
end

%% T
if ismember('T', plot_list)
    f = figure();
    axis = gca;
    for T = Ts
        err = results.(sprintf('T%d', T)).err;
        if strcmp(opts.policy, 'off')
            err = err(1:2:end,1:min(opts.epochs,10));
        end
        varplot(err(1:Ts(end)/T:end,:), 'marker', 'none', 'displayname', sprintf('%d', T))
        hold on
        axis.Children(1).EdgeColor = 'none';
        axis.Children(1).FaceAlpha = 0.2;
        axis.Children(1).HandleVisibility = 'off';
    end
    axis.XLim = [1, 10];
    axis.YScale = 'log';
    legend
    title(legend,'$T$');
    xlabel(sprintf('iterations ($kT / %d$)', opts.TK/10), 'FontSize', 20)
    ylabel('MSE', 'FontSize', 20)
    if save_flag
        exportgraphics(gca, sprintf('../fig/rr/t_err_%s.png', opts.policy), 'Resolution', 900)
        savefig(sprintf('../fig/rr/t_err_%s.fig', opts.policy))
    end
    %
    f = figure();
    axis = gca;
    for T = Ts
        expl = results.(sprintf('T%d', T)).expl;
        if strcmp(opts.policy, 'off')
            expl = expl(1:2:end,:);
        end
        varplot(expl(1:Ts(end)/T:end,:), 'marker', 'none', 'displayname', sprintf('%d', T))
        hold on
        axis.Children(1).EdgeColor = 'none';
        axis.Children(1).FaceAlpha = 0.2;
        axis.Children(1).HandleVisibility = 'off';
    end
    axis.YScale = 'log';
    legend
    title(legend,'$T$');
    xlabel(sprintf('iterations ($kT / %d$)', opts.TK/10), 'FontSize', 20)
    ylabel('Exploitability', 'FontSize', 20)
    if save_flag
        exportgraphics(gca, sprintf('../fig/rr/t_expl_%s.png', opts.policy), 'Resolution', 900)
        savefig(sprintf('../fig/rr/t_expl_%s.fig', opts.policy))
    end
end
