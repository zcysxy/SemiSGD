T=1;
[Nx,Nt] = size(u);
x=linspace(0,1,Nx);
t=linspace(0,T,T*Nt);
%{
for i=1:T*Nx
subplot(3,8,i);
plot(x,u(:,i+2));
end
%}
[T,X]=meshgrid(t,x);
%rho=tras(5).rho(:,1:32);
%rho=tra.rho(:,1:32);
s=mesh(T,x,u(:,1:length(u(1,:))));
colormap('jet');
s.FaceColor='interp';
set(gca, 'Fontsize', 20, 'linewidth', 1)
xlabel('$t$','interpreter','latex');
%ylabel('$Path:\ 1-3-4$','interpreter','latex');
%yticks([0 1]);
%ax = gca;
%ax.YTickLabel = [num2str(ax.YTick.') repmat('  ',size(ax.YTickLabel,1),1)];
%ax.YTicklabel=['1-3','3-4'];
%yticklabels({'(3\leftarrow2)','(4\leftarrow3)'})
%ytickangle(-60)
ax=gca;
YTick = get(ax, 'YTick');
YTickLabel = get(ax, 'YTickLabel');
ylabel('$x$','interpreter','latex');
set(ax,'YTick',YTick+0);
set(ax,'YTickLabel',YTickLabel);
%ax = axes();
%yticks = get(ax,'ytick');
%set(gca,'ytick',['1-3','3-4'])
%zlabel('$\u$','interpreter','latex');
%zlim([0,0.6]);
%xlim([0,5]);
%xticks([0 1 2 3 4 5])
zlabel('$u$','interpreter','latex');
%zlim([0.2,0.5]);
