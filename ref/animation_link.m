clc;
x=linspace(0,1,length(rho(:,1)));
for t=1:length(rho(1,:))/2
    plot(x,rho(:,t),'r','lineWidth',2);
    box off;
    axis([0 1 0 1]);
    ax=gca;
    xlabel('$x$','interpreter','latex','fontsize',30);
    ylabel('$\rho$','interpreter','latex','fontsize',30);
    %ax.XTick=
    %grid on;
    pause(0.1);
end