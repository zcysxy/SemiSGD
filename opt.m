clc; close all;

S=50;T=1e2; % Space-Time
gamma = 1;
delta_t=1/S;
ini_rho = ones(S,1);
rho=zeros(S,T);
rho(:,1) = (ini_rho ./ sum(ini_rho));
rho_max = max(rho(:,1));
u=zeros(S,T);
V=zeros(S+1,T+1);
n=S;
u_hist=zeros(300,S,T);
V0=0;
V(:,T+1)=V0;

for iter=1:500  
    for t=1:length(u(1,:))
        for i =1:n
            if t>1
                if i > 1; l = i - 1; else l = S; end
                if i < S; r = i + 1; else r = 1; end
                % Update pop
                rho(i,t)=0.5 * (rho(l,t-1) + rho(r,t-1)) -...
                         0.5 * (rho(r,t-1) * u(r,t-1) -...
                                rho(l,t-1) * u(l,t-1));
            end
            u(i,t) = 0.5*(1 - rho(i,t)*S/3) + 0.2 * (sin(4*pi*i/S) + 1); %lwr
            u_hist(iter,i,t)=u(i,t);
        end
        u=squeeze(sum(u_hist, 1))/iter;
        rho_max = max(rho(:,t));
    end
end

addpath('ring_road_04_04/')
plot_3D
