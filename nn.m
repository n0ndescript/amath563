%% Simulate Lorenz system
dt=0.01; 
T=8; 
t=0:dt:T;
b=8/3; 
sig=10;
r1 = 10;
Lorenz1 = @(t,x)([ sig * (x(2) - x(1)); ...
                     r1 * x(1)-x(1) * x(3) - x(2); ...
                     x(1) * x(2) - b*x(3)]);
r2 = 28;
Lorenz2 = @(t,x)([ sig * (x(2) - x(1)); ...
                     r2 * x(1)-x(1) * x(3) - x(2); ...
                     x(1) * x(2) - b*x(3)]);
r3 = 40;
Lorenz3 = @(t,x)([ sig * (x(2) - x(1)); ...
                     r3 * x(1)-x(1) * x(3) - x(2); ...
                     x(1) * x(2) - b*x(3)]);                
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];

for j=1:100 % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz1,t,x0); 
    input=[input; y(1:end-1,:)]; 
    output=[output; y(2:end,:)]; 
    [t,y] = ode45(Lorenz2,t,x0); 
    input=[input; y(1:end-1,:)]; 
    output=[output; y(2:end,:)];
    [t,y] = ode45(Lorenz3,t,x0); 
    input=[input; y(1:end-1,:)]; 
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on 
    plot3(x0(1),x0(2),x0(3),'ro')
end


%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas'; 
net.layers{3}.transferFcn = 'purelin'; 
net = train(net,input.',output.');

%%
ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; 
    x0=y0; 
end
figure(1),
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
%%
r4 = 17;
Lorenz4 = @(t,x)([ sig * (x(2) - x(1)); ...
                     r4 * x(1)-x(1) * x(3) - x(2); ...
                     x(1) * x(2) - b*x(3)]);  
for j=1:100 % test trajectories 
    [t,y] = ode45(Lorenz4,t,x0); 
end
figure(2),
plot3(y(:,1),y(:,2),y(:,3)) 
r5 = 35;
Lorenz5 = @(t,x)([ sig * (x(2) - x(1)); ...
                     r5 * x(1)-x(1) * x(3) - x(2); ...
                     x(1) * x(2) - b*x(3)]);  
for j=1:100 % test trajectories 
    [t,y] = ode45(Lorenz5,t,x0); 
end
figure(3),
plot3(y(:,1),y(:,2),y(:,3)) 
