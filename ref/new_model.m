cells=32;%64...
Terminal=1;
tol=0.01;
u=zeros(cells,cells*Terminal);
u_hist=zeros(300,cells,cells*Terminal);
u_tmp=zeros(cells,cells*Terminal);
%travel demand at time step 0 (t=1) 
d=zeros(length(u(1,:)),1);
d(2)=1;
%Initialize speed
for t=1:length(u(1,:))
    for i =1:length(u(:,1))
        u(i,t)=0.2;
    end
end
%ini_rho=tra.rho(:,1); %tras(5).rho(:,1);
%ini_rho=tra.rho_ini; %.tras(4).rho(:,1);
ini_rho=tras(5).rho(:,1);
%rho=rho_u(u,d); %initial rho
%rho=tras(4).rho(:,1:16); 
rho=rho_u(u,ini_rho); %initial rho
c=length(rho(:,1));
delta_t=1/c; %delta_t=delta_x
T=length(rho(1,:));
V=zeros(c+1,T+1); %change this one to ring road
%initialize for V
%terminal cost for cells
V0=0;
for i =1:(c+1)
    V(i,T+1)=V0;
end
a=1;
b=1;
for iter=1:2000    
    for t=1:length(u(1,:))
       for i =1:length(u(:,1))
           %u(i,t)=(V(i,t+1)-V(i+1,t+1))/(delta_t)+1-rho(i,t);
           %u(i,t)=(V(i,t+1)-V(i+1,t+1))/(delta_t)+a;
           u(i,t)=1-rho(i,t);
           if u(i,t)<=0
                u(i,t)=0;
           end
           if u(i,t)>=1
                u(i,t)=1;
           end
           %V(i,t)=delta_t*(0.5*u(i,t)^2+rho(i,t)*u(i,t)-u(i,t))+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1);
           %V(i,t)=delta_t*(0.5*u(i,t)^2+b*rho(i,t)-a*u(i,t))+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1);
           V(i,t)=delta_t*0.5*(1-rho(i,t)-u(i,t))^2+(1-u(i,t))*V(i,t+1)+u(i,t)*V(i+1,t+1);
        end 
     end 
     for t=1:(T+1)
         V(c+1,t)=V(1,t);
     end
     u_hist(iter,:,:)=u;
     u=squeeze(sum(u_hist, 1))/iter;
     rho=rho_u(u,ini_rho); 

end