function density = rho_u(u,ini_rho) %u is speed, d is demand 
    %transition matrix 
    c=length(u(:,1));
    T=length(u(1,:));
    density=zeros(c,T);
    for t=1:length(u(1,:))
        for i =1:length(u(:,1))
            if t==1
                density(i,t)=ini_rho(i); %initial rho for each cell
            else
                if i==1 
                    n=length(u(:,1));
                    density(i,t)=density(i,t-1)+density(n,t-1)*u(n,t-1)-density(i,t-1)*u(i,t-1);
                %elseif i==length(u(:,1))
                    %density(i,t)=density(i,t-1)+(density(i-1,t-1)*u(i-1,t-1)-density(i,t-1)*u(i,t-1));
                else
                    density(i,t)=density(i,t-1)+(density(i-1,t-1)*u(i-1,t-1)-density(i,t-1)*u(i,t-1));
                end
                
            end
            
        end
    end
    
end
