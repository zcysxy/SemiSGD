if ~(exist('tras', 'var'))
    load("tras.mat")
end

S=16;T=500; % Space-Time
gamma = 1;
delta_t=1/S;
ini_rho = tras(4).rho(:,1);
u=zeros(S,T);
rho=zeros(S,T);
rho(:,1) = (ini_rho ./ sum(ini_rho))*1;
V=zeros(S+1,T+1);
n=length(u(:,1));
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
                %{
                if i==1
                    rho(i,t)=rho(i,t-1)+rho(n,t-1)*u(n,t-1)-rho(i,t-1)*u(i,t-1); %use LF scheme not the old one
                else
                    rho(i,t)=rho(i,t-1)+(rho(i-1,t-1)*u(i-1,t-1)-rho(i,t-1)*u(i,t-1));
                end
                %}
            end
            u(i,t)=(V(i,t+1)-gamma * V(i+1,t+1))/(delta_t)+1-rho(i,t) + 0.1 * sin(2*pi*i/S); %lwr
            % u(i,t)=(V(i,t+1)-V(i+1,t+1))/(delta_t)+1-rho(i,t); %non-sep
            % u(i,t)=(V(i,t+1)-V(i+1,t+1))/(delta_t)+1; %sep
            % u(i,t)=(V(i,t+1)-V(i+1,t+1))/(delta_t)+1-rho(i,t); %lwr
            u(i,t) = max(0, min(u(i,t),1));
            u_hist(iter,i,t)=u(i,t);
            % Calculate cost
            %V(i,t)=delta_t*(0.5*u(i,t)^2+rho(i,t)*u(i,t)-u(i,t))+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1);
            % V(i,t)=delta_t*(0.5*u(i,t)^2+rho(i,t)-u(i,t))+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1);  %sep
            V(i,t)=delta_t*0.5*(1-rho(i,t) + 0.0 * sin(2*pi*i/S) -u(i,t))^2+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1); %lwr
        end
        V(S+1,t)=V(1,t);
        u=squeeze(sum(u_hist, 1))/iter;
    end
end

plot_3D