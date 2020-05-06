years = 1845:2:1903;
years = (years-1845)';
hare = [20;20;52;83;64;68;83;12;36;150;110;60;7;10;70;100;92;70;10;11;137;137;18;22;52;83;18;10;9;65];
lynx = [32;50;12;10;13;36;15;12;6;6;65;70;40;9;20;34;45;40;15;15;60;80;26;18;37;50;35;12;12;25]; 


dt = years(2) - years(1);
X = [hare,lynx]';
X1 = X(:,1:end-1); 
X2 = X(:,2:end);
r=2;
[U2,Sigma2,V2] = svd(X1,'econ'); 
plot(diag(Sigma2), (sum(diag(Sigma2))), 'ro', 'Markersize', 10, ...
                                    'MarkerEdgeColor','k',...
                                    'MarkerFaceColor',[.49 1 .63]);
U=U2(:,1:r); 
Sigma=Sigma2(1:r,1:r); 
V=V2(:,1:r);
 
% DMD J-Tu decomposition:  Use this one
    
Atilde = U'*X2*V/Sigma;    
[W,D] = eig(Atilde);    
Phi = X2*V/Sigma*W;
    
mu = diag(D);
omega = log(mu)/dt;

u0 = X(:,1);
y0 = Phi\u0;  % pseudo-inverse initial conditions
u_modes = zeros(r,length(years));
for iter = 1:length(years)
     u_modes(:,iter) =(y0.*exp(omega*years(iter)));
end
u_dmd = Phi*u_modes;
    


% DMD doesn't give us a reasonable answer here so let's try time delay
% embedding to capture non linear dynamics 

H = [hare(1:20).'
     lynx(1:20).'
     hare(2:21).'
     lynx(2:21).'
     hare(3:22).'
     lynx(3:22).'
     hare(4:23).'
     lynx(4:23).'
     hare(5:24).'
     lynx(5:24).'
     hare(6:25).'
     lynx(6:25).'
     hare(7:26).'
     lynx(7:26).'
     hare(8:27).'
     lynx(8:27).'
     hare(9:28).'
     lynx(9:28).'
     hare(10:29).'
     lynx(10:29).'
     hare(11:30).'
     lynx(11:30).'];

 
X1 = H(:,1:end-1); 
X2 = H(:,2:end);
r=15;
[U2,Sigma2,V2] = svd(X1,'econ'); 
plot(diag(Sigma2), (sum(diag(Sigma2))), 'ro', 'Markersize', 10, ...
                                    'MarkerEdgeColor','k',...
                                    'MarkerFaceColor',[.49 1 .63]);
U=U2(:,1:r); 
Sigma=Sigma2(1:r,1:r); 
V=V2(:,1:r);
 
% DMD J-Tu decomposition:  Use this one
    
Atilde = U'*X2*V/Sigma;    
[W,D] = eig(Atilde);    
Phi = X2*V/Sigma*W;
    
mu = diag(D);
omega = log(mu)/dt;

u0 = H(:,1);
y0 = Phi\u0;  % pseudo-inverse initial conditions
u_modes = zeros(r,length(years));
for iter = 1:length(years)
     u_modes(:,iter) =(y0.*exp(omega*years(iter)));
end
u_dmd_time_delay = Phi*u_modes;

figure(2)
plot(real(u_dmd_time_delay(1,1:30)), 'Color', 'red', 'Linewidth', 2 )
hold on
plot(hare,'Color', 'blue', 'Linewidth', 2)

figure(3) 
plot(real(u_dmd(1,1:30)), 'Color', 'magenta', 'Linewidth', 2 )
hold on
plot(hare,'Color', 'blue', 'Linewidth', 2)

figure(4) 
plot(real(u_dmd_time_delay(2,1:30)), 'Color', 'red', 'Linewidth', 2 )
hold on
plot(lynx,'Color', 'green', 'Linewidth', 2)

figure(5) 
plot(real(u_dmd(2,1:30)), 'Color', 'magenta', 'Linewidth', 2 )
hold on
plot(lynx,'Color', 'green', 'Linewidth', 2)

%% Solving the Lotka Volterra Equation Numerically 
td = [1:30];
hare = [20;20;52;83;64;68;83;12;36;150;110;60;7;10;70;100;92;70;10;11;137;137;18;22;52;83;18;10;9;65]';
lynx = [32;50;12;10;13;36;15;12;6;6;65;70;40;9;20;34;45;40;15;15;60;80;26;18;37;50;35;12;12;25]';
p = [20 32 0.5, 0.028 0.84 0.026]; % Initial values 


[p,fval,exitflag] = fminsearch(@leastcomp,p,[],td,hare,lynx);
t0 = 1;
tfinal = 30;
y0 = [p(1); p(2)];   
[t,y_lv] = ode23(@lotka,tspan,y0, [], p(3),p(4),p(5),p(6));
%[t,y] = ode23(@lotvol,[t0 tfinal],y0,[], p(3),p(4),p(5),p(6));

%% Model Estimation Using Sparse Regression
dt=2;
t=1:dt:60; 
x0=[0.1 5];
mu=1.2;

x1=hare';
x2=lynx';

n=length(t);
for j=2:n-1
  x1dot(j-1)=(x1(j+1)-x1(j-1))/(2*dt);
  x2dot(j-1)=(x2(j+1)-x2(j-1))/(2*dt);
end

x1s=x1(2:n-1);
x2s=x2(2:n-1);
A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x2s.^2).*x1s x2s.^3 sin(x1s) sin(x2s) ];
%A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x1s.^2).*x2s (x2s.^2).*x1s x2s.^3];

xi1=A\x1dot.';
xi2=A\x2dot.';
subplot(2,1,1), bar(xi1)
subplot(2,1,2), bar(xi2)

%% KL Divergence 
n=30;
x1=hare; % 
x2=u_dmd(1,:); % DMD with no time delay
x3=real(u_dmd_time_delay(1,:)); % DMD with time delay
x4= y_lv(:,1).'; % Solving the Lotka Volterra Numerically 

n=30;
x1=lynx; % 
x2=u_dmd(2,:); % DMD with no time delay
x3=real(u_dmd_time_delay(2,:)); % DMD with time delay
x4= y_lv(:,2).'; % Solving the Lotka Volterra Numerically 

x=1:1:100; % range for data
f=hist(x1,x)+0.01; % generate PDFs
g1=hist(x2,x)+0.01;
g2a=hist(x3,x); g2b=hist(x4,x); g2=g2a+0.3*g2b+0.01;
g3=hist(x5,x)+0.01;
f=f/trapz(x,f); % normalize data
g1=g1/trapz(x,g1); g2=g2/trapz(x,g2); g3=g3/trapz(x,g3);
plot(x,f,x,g1,x,g2,x,g3,'Linewidth',[2])
% compute integrand
Int1=f.*log(f./g1); Int2=f.*log(f./g2); Int3=f.*log(f./g3);
% use if needed
%Int1(isinf(Int1))=0; Int1(isnan(Int1))=0;
%Int2(isinf(Int2))=0; Int2(isnan(Int2))=0;
% KL divergence
I1=trapz(x,Int1); I2=trapz(x,Int2); I3=trapz(x,Int3);

%%

function J = leastcomp(p,tdata,xdata,ydata)
%Create the least squares error function to be minimized.
    n1 = length(tdata);
    [~,y] = ode23(@lotka,tdata,[p(1),p(2)],[],p(3),p(4),p(5),p(6));
    errx = y(:,1)-xdata(1:n1)';
    erry = y(:,2)-ydata(1:n1)';
    J = errx'*errx + erry'*erry;
 
end

function dydt = lotvol(~,y,a1,a2,b1,b2)
    % Predator and Prey Model
    tmp1 = a1*y(1) - a2*y(1)*y(2);
    tmp2 = -b1*y(2) + b2*y(1)*y(2);
    dydt = [tmp1; tmp2];
end

function yp = lotka(t,y,a1, a2, b1, b2)
%LOTKA  Lotka-Volterra predator-prey model.

%   Copyright 1984-2014 The MathWorks, Inc.

yp = diag([a1 - a2*y(2), -b1 + b2*y(1)])*y;
end


