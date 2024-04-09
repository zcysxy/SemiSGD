c=length(rho(:,1));
T=length(rho(1,:));
data_rho=zeros(c+1,T+1);
data_u=zeros(c+1,T+1);
data_V=zeros(c+1,T+2);
for t=1:length(data_rho(1,:))
    for i=1:length(data_rho(:,1))
        if t==1
            data_rho(i,t)=1/8*(i-2);
            data_u(i,t)=1/8*(i-2);
            data_V(i,t)=1/8*(i-2);
        elseif i==1
            data_rho(i,t)=1/8*(t-2);
            data_u(i,t)=1/8*(t-2);
            data_V(i,t)=1/8*(t-2);
        else
            data_rho(i,t)=rho(i-1,t-1);
            data_u(i,t)=u(i-1,t-1);
            data_V(i,t)=V(i-1,t-1);
        end
    end
end
data_rho(1,1)=0;
data_u(1,1)=0;
data_V(1,1)=0;
data_V(1,34)=1;
csvwrite('data_rho_new.csv',data_rho)
%csvwrite('data_u_sep_new.csv',data_u)
%csvwrite('data_V_sep_new.csv',data_V)
%csvwrite('data_u.csv',data_rho)

