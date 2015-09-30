%% Fixing problems with Drawing procedure
S = SetSystem;
[x, y, z] = cylinder(S.R,50);
plot([-S.R-S.R/5 S.R + S.R/5],[0,0],'--','linewidth',1);hold on; grid on;
plot([0,0],[-S.R-S.R/5 S.R + S.R/5],'--','linewidth',1);
plot(x(1,:),y(1,:));
xl = S.R - S.g;
plot([xl xl],[-S.D/2 S.D/2],'m')
l = sqrt(S.R^2-S.D^2/4);
plot([l l],[-S.D/2 S.D/2],'m')
alp = atan(S.D/2/l);
al = pi/2+linspace(-alp,alp,20);
xa = sin(al)*S.R;
ya = cos(al)*S.R;
plot(xa,ya,'m','linewidth',3)
hold off;
%% Drawing of proper system
S = SetSystem;
% S.D = 13;
% S.R = 10.24;
% S.tc = 2;
DrawLensSystem(S);
S.Cs1  = S.l1 - S.R + S.g; 
S.Cs2  = S.Cs1 + S.ll + 2*S.R;
hold on;
plot3(S.Cs1,0,0,'p','color','r');  
  text(S.Cs1,1,0,'Cs1');
plot3(S.Cs2,0,0,'p','color','r');  
  text(S.Cs2,1,0,'Cs2');
  [x,y,z] = cylinder(S.R,50 );
plot3(x(1,:)+S.Cs1,y(1,:),z(1,:))  
plot3(x(1,:)+S.Cs2,y(1,:),z(1,:))  


%% Calculating trace for ray correspond to the border of diaphragm 
%   clear all; close all;

S = SetSystem;
S.lambda = 400;
S.D = 12;
S.lCCD = 82.0;
S.l1 = 20;

S1 = S;
S1.lambda = 800;

DrawLensSystem(S)
hold on;

Dw = linspace(-S.dW/2,S.dW/2,5);
Dh = linspace(-S.dH/2,S.dH/2,5);
%
for i = 1 : length(Dw)
    for j = 1 : length(Dh)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % toczka na kraju diafragmy
        P = RayTracing(Pd,S);
        Pd1 = [ S1.ld, Dw(i), Dh(j) ]; % toczka na kraju diafragmy
         P1 = RayTracing(Pd1,S1);
        plot3(P(:,1),P(:,2),P(:,3))
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r')
        
        plot3(P1(:,1),P1(:,2),P1(:,3),'color','m')
        plot3(P1(:,1),P1(:,2),P1(:,3),'p','color','r')

    end
end

Dw = linspace(-S.dW/2,S.dW/2,2);
Dh = linspace(-S.dH/2,S.dH/2,10);
for i = 1 : length(Dw)
    for j = 1 : length(Dh)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % toczka na kraju diafragmy
        P = RayTracing(Pd,S);
        P1 = RayTracing(Pd,S1);
        plot3(P(:,1),P(:,2),P(:,3))
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r')
        
        plot3(P1(:,1),P1(:,2),P1(:,3),'color','m')
        plot3(P1(:,1),P1(:,2),P1(:,3),'p','color','m')
    end
end
set(gca,'fontsize',14)
xlabel('X','fontsize',14);
ylabel('Y','fontsize',14);
zlabel('Z','fontsize',14);
% hold off;
% view([1,52])
% figure

%% Look for function which bind Y_in and Y_out
% Y -  coordinate is binded to the width ( W ) of diaphragm 
S = SetSystem;
DrawLensSystem(S)
S.D = 15;
hold on;

Dw = linspace(-S.dW/2,S.dW/2,10);
Dh = [-1 -.5 0 .5 1];
Pd = [ S.ld, 0, 0 ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
         plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
%
figure;
Yout = zeros(5,length(Dw));
for j = 1:length(Dh)
for i = 1 : length(Dw)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
        Yout(j,i)=P(7,2);
        plot3(P(:,1),P(:,2),P(:,3));
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
   
end
end
hold off;
view([1,0,0])
% figure; 
plot(Dw,Yout(5,:))
%% Calculation  for the equally spaced rays 
S = SetSystem;
DrawLensSystem(S)
S.D = 15;
hold on;

Dw = linspace(-S.dW/2,S.dW/2,10);
Dh = linspace(-S.dH/2,S.dH/2,10);
for i = 1 : 1%length(Dw)
    for j = 1 : 1%length(Dh)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
        plot3(P(:,1),P(:,2),P(:,3));
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
    end
end
hold off;
view([1,0,0])
%% Calculation  for the equally spaced rays 
S = SetSystem;
S.D = 15;
DrawLensSystem(S)

hold on;
Dw = linspace(-S.dW/2,S.dW/2,10);
Dh = linspace(-S.dH/2,S.dH/2,10);
inMatrix = meshgrid(Dw,Dh);
inX = meshgrid(Dw);
surf(Dw,Dh,inX);
%
outX = zeros(length(Dw),length(Dh));
outY = outX;
outZ = outX;
for i = 1 : length(Dw)
    for j = 1: length(Dh)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
        plot3(P(:,1),P(:,2),P(:,3));
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
        outX(i,j) = P(7,1);
        outY(i,j) = P(7,2);
        outZ(i,j) = P(7,3);
    end
end
%% Checking the calculation of the lens system
% Pinhole at the distance of l = 17 mm from first lens;
% TODO: ! Narisowat rasklad intensiwnosti !
l = 17.9;
sx = 0;
sy = 0;
N_points = 50;
[x,y,z] = cylinder(1.7,N_points);
P1 = [z(1,:);y(1,:);x(1,:)];
P2 = [z(1,:)+l;y(1,:)+sy; x(1,:)+sx];
% % making the cylinder
% plot3(P1(1,:),P1(2,:),P1(3,:),'g');hold on; grid on;
% plot3(P2(1,:),P2(2,:),P2(3,:),'r'); 
% % direction vector of the line
% nV = P2-P1;
% % here we  created our beams 
% quiver3(P1(1,:),P1(2,:),P1(3,:),nV(1,:),nV(2,:),nV(3,:),0)
% hold off;
% creating of lens system 
S = SetSystem;
S.lambda = 457.09;
S.ll = 37.2;
S.lCCD = 100;
subplot(2,1,1)
DrawLensSystem(S)
hold on;
C_in =[];
C_out =[];
for i = 1 : N_points 
    S.Pk = P1(:,i).';
    Pd = [ P2(1,i), P2(2,i), P2(3,i) ]; % Points on the diaphragm plane 
    P = RayTracing(Pd,S);
    plot3(P(:,1),P(:,2),P(:,3));
    C_in(end+1,1:2) = [P(1,2),P(1,3)];
     C_out(end+1,1:2) = [P(7,2),P(7,3)];
end
hold off;
subplot(2,1,2)
plot(C_in(:,1),C_in(:,2),'b',...
     C_out(:,1),C_out(:,2),'r')
%% Comparison of angle before and after lens system
S = SetSystem;
S.lambda = 457.09;
S.Pk = [0,0,0];
DrawLensSystem(S);  hold on;
 Pd = [S.ld, 0, S.dH/2 ]; % Points on the diaphragm plane 
 P = RayTracing(Pd,S);
 plot3(P(:,1),P(:,2),P(:,3),'linewidth',3); 
 Pd = [S.ld, 0, 0 ]; % Points on the diaphragm plane 
 P0 = RayTracing(Pd,S);
plot3(P0(:,1),P0(:,2),P0(:,3),'.-','color','r','linewidth',3); hold off;
set(gca,'view',[0,0])
% Angle calculation
Pd = [S.ld, 0, S.dH/2 ];
n1 = (Pd - S.Pk)/norm(Pd - S.Pk); % direction vector
n  = [1, 0, 0 ]; % direction vector of line (optical axis)
alpha1 = acosd( dot(n1,n)); % angle before lens system
% atand(S.dH/2/(S.Pk(1)-S.ld)); % checking
n2 = (P(7,:)-P(6,:))/norm(P(7,:)-P(6,:));
alpha2 = acosd( dot(n2,n));
% angle after lens system
figure;
l = 10;
plot([0,l],[0 2*l*tand(alpha1)],...
     [0,l],[0 2*l*tand(alpha2)],...
     [l,l],[2*l*tand(alpha1),2*l*tand(alpha2)]);
 xlim([0,l+l/20]);
grid on;
s1 = sprintf('Ray before lens system \n %s %f [deg]','\alpha =',alpha1);
s2 = sprintf('Ray after lens system \n %s %f [deg]','\alpha =',alpha2);
s3 = sprintf('deviation  %s %f [mm]','\Delta = ',2*l*(tand(alpha1)-tand(alpha2)));
legend(s1,s2,s3,2);
xlabel('x[ mm ]')
ylabel('y[ mm ]')
%% FIXME: Fix error: border of aperture is not straight
% draw electrode
H = 4.22; % height of aperture
W = 9.00; % widht of aperture
Dout = 29.64; % outer diameter 
Din = 23.69;  % inner diameter
[ x, y, z ] = cylinder(Dout/2,50);
plot3(x(1,:),y(1,:),ones(1,size(x,2))*-H/2,...
    x(1,:),y(1,:),ones(1,size(x,2))*H/2,'b'); hold on; grid on;
%
[ x, y, z ] = cylinder(Din/2,50);
plot3(x(1,:),y(1,:),ones(1,size(x,2))*-H/2,'r',...
      x(1,:),y(1,:),ones(1,size(x,2))*H/2,'r');
% border of aperture
Rin = Din/2; % inner radius
alpha = asin(W/2/Rin);
[X,Y] = pol2cart(alpha,Rin);
plot3(X,Y,z(2,:)*H/2,'p'); 
[X,Y] = pol2cart(-alpha,Rin);
plot3(X,Y,z(2,:)*H/2,'p'); 

alpha_i = linspace(-alpha,alpha,20);
    [X,Y] = pol2cart(alpha_i,Rin);
    plot3(X,Y,ones(1,length(alpha_i))*H/2,'p');
    plot3(X,Y,-ones(1,length(alpha_i))*H/2,'p'); 
% outer electrode
Rout = Dout/2;
alpha = asin(W/2/Rout);
[X,Y] = pol2cart(alpha,Rout);
plot3(X,Y,z(2,:)*H/2,'p'); 
[X,Y] = pol2cart(-alpha,Rout);
plot3(X,Y,z(2,:)*H/2,'p'); 

%% Creation of border line
% parameters of electrode
Br = [];
H = 4.22; % height of aperture
W = 9.00; % widht of aperture
Dout = 29.64; % outer diameter 
Din = 23.69;  % inner diameter
N = 25;       % number of points
% The beginning of the coordinate system is placed at the center of the trap
alpha = asin(W/Dout); % angle to the left or right corner
% left and right border
[X,Y] = pol2cart(alpha,Dout/2);
Vx = ones(1,N)*X;
Vy = ones(1,N)*Y;
Vz = linspace(-H/2,H/2,N);
% right corner
[X,Y] = pol2cart(-alpha,Dout/2);
Br(:,1) = [Vx ones(1,N)*X];
Br(:,2) = [Vy ones(1,N)*Y];
Br(:,3) = [Vz linspace(-H/2,H/2,N)];
plot3(Br(:,1),Br(:,2),Br(:,3),'*','color','r') 
alpha_i = linspace(-alpha,alpha,20);
% position of droplet
Pd = [0,0,0]; 
 text(Pd(1),Pd(2),Pd(3),'Pd');
    plot3(Pd(1),Pd(2),Pd(3),'*')
% Creation of vector
% for i = 1:size(Br,1)
%     V = Br(i,:) - Pd;
%     % normalization
%     V = V/norm(V);
%     t = [0,20];
%    
%     plot3(Pd(1)+V(1)*t,Pd(2)+V(2)*t,Pd(3)+V(3)*t,'linewidth',2)
% end

% intersection with the inner circle

V = Br(5,:) - Pd; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*Pd(1)+V(2)*Pd(2));
C = Pd(1)^2+Pd(2)^2 - (Din/2)^2;
D = B^2-4*A*C;
t =[0 (-B+sqrt(D) )/2/A];
% plot(Pd(1)+V(1)*t,Pd(2)+V(2)*t,'*','color','r')
plot3(Pd(1)+V(1)*t,Pd(2)+V(2)*t,Pd(3)+V(3)*t,'linewidth',2,'color','r')

Pb(1,1) = Pd(1)+V(1)*t(end);
Pb(1,2) = Pd(2)+V(2)*t(end);

V = Br(end,:) - Pd; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*Pd(1)+V(2)*Pd(2));
C = Pd(1)^2+Pd(2)^2 - (Din/2)^2;
D = B^2-4*A*C;
t =[0 (-B+sqrt(D) )/2/A];
% plot(Pd(1)+V(1)*t,Pd(2)+V(2)*t,'*','color','r')
Pb(2,1) = Pd(1)+V(1)*t(end);
Pb(2,2) = Pd(2)+V(2)*t(end);

plot3(Pd(1)+V(1)*t,Pd(2)+V(2)*t,Pd(3)+V(3)*t,'linewidth',2,'color','r')
plot3([0,Pb(1,1)],[0 Pb(1,2)],[0,0],'linewidth',2,'color','g');
plot3([0,Pb(2,1)],[0 Pb(2,2)],[0,0],'linewidth',2,'color','g')

Bt1 = acos( dot(Pb(1,:),[1,0])/norm(Pb(1,:)));
Bt2 = acos( dot(Pb(2,:),[1,0])/norm(Pb(2,:)));
VBt = linspace(Bt1,-Bt2,30);
[X,Y] = pol2cart(VBt,Din/2);
plot3([0 10],[0 0],[0,0])
plot3(X,Y,zeros(size(X)),'o','color','m')
hold off;
set(gca,'view',[0 90]);
%% Done: Creation of electrode Borders 
clf;
S = SetSystem;
S.Rin = 14.82;
S.Rout = 29.68/2;
S.Pk = [0,0,0];
% Drawing for proof
Np = 50;
% outer ellectrodes
[xc, yc ] = cylinder(S.Rout,Np);
 plot3(xc(1,:),yc(1,:),ones(1,length(xc))*S.dH/2); hold on;
 plot3(xc(1,:),yc(1,:),-ones(1,length(xc))*S.dH/2); 
% inner electrodes 
 [xc, yc ] = cylinder(S.Rin,Np);
 plot3(xc(1,:),yc(1,:),ones(1,length(xc))*S.dH/2); hold on;
 plot3(xc(1,:),yc(1,:),-ones(1,length(xc))*S.dH/2); 
% droplet position 
text(S.Pk(1),S.Pk(2),S.Pk(3),'D')
plot3(S.Pk(1),S.Pk(2),S.Pk(3),'*')

Br = zeros(4*S.N,3);       % vector of border points
% calculation 4 outer points
alpha = asin(S.dW/2/S.Rout);
[X(1),Y(1)] = pol2cart(alpha,S.Rout);
[X(2),Y(2)] = pol2cart(-alpha,S.Rout);
Z(1) = -S.dH/2;
Z(2) = S.dH/2;
P(1,:) = [X(1),Y(1),Z(1)];
P(2,:) = [X(1),Y(1),Z(2)];
P(3,:) = [X(2),Y(2),Z(2)];
P(4,:) = [X(2),Y(2),Z(1)];
% intersection with the inner circle
% lines from droplet to borders
plot3([S.Pk(1) P(1,1)],[S.Pk(2) P(1,2)],[S.Pk(3) P(1,3)])
plot3([S.Pk(1) P(4,1)],[S.Pk(2) P(4,2)],[S.Pk(3) P(4,3)])
plot3([S.Pk(1) P(2,1)],[S.Pk(2) P(2,2)],[S.Pk(3) P(2,3)])
plot3([S.Pk(1) P(3,1)],[S.Pk(2) P(3,2)],[S.Pk(3) P(3,3)])


V = P(4,:) - S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*S.Pk(1)+V(2)*S.Pk(2));
C = S.Pk(1)^2+S.Pk(2)^2 - (S.Rin)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(5,:) = [ S.Pk(1)+V(1)*t S.Pk(2)+V(2)*t -S.dH/2];

V = P(1,:) - S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*S.Pk(1)+V(2)*S.Pk(2));
C = S.Pk(1)^2+S.Pk(2)^2 - (S.Rin)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(6,:) = [ S.Pk(1)+V(1)*t S.Pk(2)+V(2)*t -S.dH/2];

P(7,1:2) = P(6,1:2);
P(7,3)   = S.dH/2;

P(8,1:2) = P(5,1:2);
P(8,3)   = S.dH/2;

plot3(P(:,1),P(:,2),P(:,3),'*','color','m'); hold on;

for i = 1 : size(P,1)
text(P(i,1),P(i,2),P(i,3),['P',num2str(i)])
end
%
Br(1:S.N,1) = ones(1,S.N)*P(1,1);
Br(1:S.N,2) = ones(1,S.N)*P(1,2);
Br(1:S.N,3) = linspace(P(2,3),P(1,3),S.N);
V1 = P(6,:); % directing vector from origin to point 6
V2 = P(5,:); % directing vector from origin to point 5
Bt1 = acos( dot( V1(1:2) ,[1,0])/norm( V1(1:2) ));
Bt2 = acos( dot(V2(1:2),[1,0])/norm(V2(1:2)));
VBt = linspace(Bt1,-Bt2,S.N);
%
[X,Y] = pol2cart(VBt,S.Rin);
plot3(Br(:,1),Br(:,2),Br(:,3),'p','color','r'); 
plot3(X,Y,-ones(1,S.N)*S.dH/2,'.');

Br((S.N+1):2*S.N,1) = X;
Br((S.N+1):2*S.N,2) = Y;
Br((S.N+1):2*S.N,3) = -ones(1,S.N)*S.dH/2;

Br((2*S.N+1):3*S.N,1) = ones(1,S.N)*P(4,1);
Br((2*S.N+1):3*S.N,2) = ones(1,S.N)*P(4,2);
Br((2*S.N+1):3*S.N,3) = linspace(P(4,3),P(3,3),S.N);

VBt = linspace(-Bt2,Bt1,S.N);
%
[X,Y] = pol2cart(VBt,S.Rin);
Br((3*S.N+1):4*S.N,1) = X;
Br((3*S.N+1):4*S.N,2) = Y;
Br((3*S.N+1):4*S.N,3) = ones(1,S.N)*S.dH/2;
set(gca,'view',[0 90]);

plot3(Br(:,1),Br(:,2),Br(:,3),'-P','color','m'); grid on; 
hold off;
set(gca,'fontsize',14);
xlabel('X','fontsize',14)
ylabel('Y','fontsize',14)
zlabel('Z','fontsize',14)
%% Ray trasing
S = SetSystem;
S.D = 15;
S.Rin = 14.82;
S.Rout = 29.68/2;

S.Pk = [0,0,0];
% DrawLensSystem(S)

for j = 1 : size(Br,1)
        Pd = [ Br(j,1), Br(j,2), Br(j,3) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
%         plot3(P(:,1),P(:,2),P(:,3));
%         plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
if size(P,1) == 7
    outX(j) = P(7,1);
        outY(j) = P(7,2);
        outZ(j) = P(7,3);
else
    
        outX(j) = NaN;
        outY(j) = NaN;
        outZ(j) = NaN;
end
        
    end

plot(outY,outZ,'linewidth',2); grid on; hold on
% recalculation of diaphragm position
Rt = 29.64/2; % radius of middle electrode
al = asin(S.dW/2/Rt);
S.ld = Rt*cos(al); % proper position of the first diaphragm
S.Pk = [0,0,0];
% S.lCCD = 92.6;
%
vW = [ linspace(-S.dW/2,S.dW/2,S.N),...
       ones(1,S.N)*S.dW/2,...
       linspace(S.dW/2, -S.dW/2,S.N),...
       -1*ones(1,S.N)*S.dW/2 ];
%   
vH =  [ ones(1,S.N)*S.dH/2,...
         linspace(S.dH/2,-S.dH/2,S.N),...
         -1*ones(1,S.N)*S.dH/2,...
         linspace(-S.dH/2, S.dH/2,S.N) ];
     
for i = 1:length(vW)

         Pd = [ S.Pk(1)+S.ld, vW(i), vH(i) ];
         
         P = RayTracing(Pd,S);
         
             if size(P,1) == 7
                 W1(i) = P(end,2);
                 Hi1(i) = P(end,3);
             else
                 
                 W1(i) = NaN;
                 Hi1(i) = NaN;
             end
 end
plot(W1,Hi1,'color','r','linewidth',2);
xlabel('X','fontsize',14);
ylabel('Y','fontsize',14);
hold off;
%% Comparison with Oslo
S = SetSystem;
S.lambda = 476;
S.Rin = 14.82;
S.Rout = 29.68/2;

DrawLensSystem(S);
S.Pk = [-.4 ,0,0];
hold on;
Dw = linspace(-S.dW/2,S.dW/2,12);
Dh = linspace(-S.dH/2,S.dH/2,12);
%
outX = zeros(length(Dw),length(Dh));
outY = outX;
outZ = outX;
for i = 1 : length(Dw)
    for j = 1: length(Dh)
        Pd = [ S.ld, Dw(i), Dh(j) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,S);
        plot3(P(:,1),P(:,2),P(:,3));
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
        outX(i,j) = P(7,1);
        outY(i,j) = P(7,2);
        outZ(i,j) = P(7,3);
    end
end





%% Reading frames from video
fn = 'I:\!From Justice\0.1ml\Splited\DEG_0.1mlSio21_1804101401.avi';
[mov]=AviReadPike_Split(fn,1:10);
size(mov);
i = 10;
imshow(squeeze(mov(i,:,:,1)),[0,2e3]);
R = squeeze( mov(i,:,:,1));
G = squeeze( mov(i,:,:,2));
B = squeeze( mov(i,:,:,3));
RGB(:,:,1) = R;
RGB(:,:,2) = G;
RGB(:,:,3) = B;
imshow(uint16(RGB).*4)
%% Red chanel
imshow(uint16(R).*5);
hold on;
plot([1,640],[200 200]);
hold off

%% Plotting data for R,G,B, channels
plot(R(200,:)*9,'r');  hold on;
plot(G(200,:),'g');
plot(B(200,:),'b');
hold off;
%%  check movie
fn = 'I:\!From Justice\0.1ml\Splited\DEG_0.1mlSio21_1804101401.avi';
inf = aviinfo(fn);
for i = 1: inf.NumFrames
    [mov]=AviReadPike_Split(fn,i);
    RGB = uint16( squeeze(mov) ).*5;
    imshow(RGB);
    title(num2str(i))
    drawnow;
end
%% Lenard Djons potential
r = 370:7e2;
k1 = 1;
k2 = 1;
Rinkl = 225;
LD = ( k1*( 2*Rinkl ./ r ).^12 - k2*2*(2*Rinkl./r).^6);
plot(r,LD)
%% Pixel size determination
% angle      - [rad]
% distances  - [m]
theta = -10:.01:10;
d = 3e-5;
lambda = 457e-9;
% intensity - [arb.U]
I0 = 4095;
x =( d * pi * sind(theta) / lambda );
% angle destributing of intensity
I = I0*(sin(x)./x).^2;
plot( theta,I )
% recalculation to pixels
ld = 1.5e-2;    % [m] 
Psize = 1e-5;  % [m]
p = tand(theta)*ld/Psize;
plot(p,I,'linewidth',2); grid on;
hold on;
ld = 1.5e-2;    %[m] 
p = tand(theta)*ld/Psize;
plot(p,I,'linewidth',2,'color','r');
hold off;
xlabel('W[ Pix ]')
xlim([-320 320])
%% Diffraction on slit  Comparison with CCD matrix
P  = -320:320;   % CCD detector in [ Pix ]
Ps = .74e-5;     % size of one pixel [ m ]
L  = 5.4e-2;      % distance between slit and CCD [ m ]
Theta = atan(P*Ps/L); % angle range [rad]
d  = 3e-5;     % width of slit  [m]
lambda = 457e-9;   % wavelength [m]
I0 = 1;    % max sensitivity of CCD [arb.U]
% ------
X  = d*pi*sin(Theta)/lambda;
I = I0*(sin(X)./X).^2;
plot(320+P,I,'linewidth',2); grid on; hold on;
% ------
lambda = 658e-9;

X  = d*pi*sin(Theta)/lambda;
I = I0*(sin(X)./X).^2;
plot(320+P,I,'linewidth',2,'color','r'); 

xlabel('W[ Pix ]');
xlim([0 640]);
hold off;
%% --- Dif on slit: prepearing Exp. data
% % loading dta
%    clear;
%    cd('D:\!Work\!Projects\Aberration correction');
%    load('First_Position.mat')
%  microscope 49 dash - 0.24 [mm] = 240[mkm]
%  1 dash = 4.8980 [mkm]
%  d - slit = 9 dash = 44.1 [mkm]

%% --- Dif on slit: W1 L2
    Im = mean(w1_l2mm,3);
        I = mean(Im,1);
        I(640) = I(639); 
        I = I-min(I(:));
        I1 = I./(max(I));
    plot(I1,'b'); hold on;
      ylim([0,0.1]);
   [x,y] = getpts;
      Bg = fit(x,y,'smooth');
    plot(Bg,'r');
      W_L2(:,1) = I1(:) - Bg(1:640);
    plot(W_L2,'g');
    hold off;
%% --- Dif on slit: W2 L2
j = 2;
  Im = mean(w2_l2mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L2(:,j) = I1(:) - Bg(1:640);
  plot(W_L2(:,j),'g');
    hold off;
%% --- Dif on slit: W3 L2
   j = 3;
  Im = mean(w3_l2mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L2(:,j) = I1(:) - Bg(1:640);
  plot(W_L2(:,j),'g');
    hold off;
    
%% --- Dif on slit:  W4 L2
   j = 4;
  Im = mean(w4_l2mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L2(:,j) = I1(:) - Bg(1:640);
  plot(W_L2(:,j),'g');
    hold off;
    
%% --- Dif on slit:  W5 L2
     j = 5;
  Im = mean(w5_l2mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L2(:,j) = I1(:) - Bg(1:640);
  plot(W_L2(:,j),'g');
    hold off;

%% --- Dif on slit: W1 L5
     j = 1;
  Im = mean(w1_l5mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L5(:,j) = I1(:) - Bg(1:640);
  plot(W_L5(:,j),'g');
    hold off;
    
%% --- Dif on slit: W2 L5
    j = 2;
  Im = mean(w2_l5mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L5(:,j) = I1(:) - Bg(1:640);
  plot(W_L5(:,j),'g');
    hold off;
    
%% --- Dif on slit: W3 L5
   j = 3;
  Im = mean(w3_l5mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L5(:,j) = I1(:) - Bg(1:640);
  plot(W_L5(:,j),'g');
    hold off;
%% --- Dif on slit: W4 L5
    j = 4;
  Im = mean(w4_l5mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L5(:,j) = I1(:) - Bg(1:640);
  plot(W_L5(:,j),'g');
    hold off;
    
%% --- Dif on slit: W5 L5
    j = 5;
  Im = mean(w5_l5mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L5(:,j) = I1(:) - Bg(1:640);
  plot(W_L5(:,j),'g');
    hold off;
    
%% --- Dif on slit: W1 L20
    j = 1;
  Im = mean(w1_l20mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L20(:,j) = I1(:) - Bg(1:640);
  plot(W_L20(:,j),'g');
    hold off;    
    
%% --- Dif on slit: W2 L20
    j = 2;
  Im = mean(w2_l20mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L20(:,j) = I1(:) - Bg(1:640);
  plot(W_L20(:,j),'g');
    hold off;        
    
%% --- Dif on slit: W3 L20
    j = 3;
  Im = mean(w3_l20mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L20(:,j) = I1(:) - Bg(1:640);
  plot(W_L20(:,j),'g');
    hold off;            
    
%% --- Dif on slit: W4 L20
    j = 4;
  Im = mean(w4_l20mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L20(:,j) = I1(:) - Bg(1:640);
  plot(W_L20(:,j),'g');
    hold off;            
    
    
%% --- Dif on slit: W5 L20
    j = 5;
  Im = mean(w5_l20mm,3);
    I = mean(Im,1);
    I(640) = I(639);
    I = I-min(I(:));
    I1 = I./(max(I));
  plot(I1,'b'); hold on;
    ylim([0,0.1]);
    [x,y] = getpts;
    Bg = fit(x,y,'smooth');
  plot(Bg,'r');
    W_L20(:,j) = I1(:) - Bg(1:640);
  plot(W_L20(:,j),'g');
    hold off;            
    
%% ---- Dif on slit Theory
% We build variable in that way:
%  W1(1,:) - W1 - number is wavelenth, first dimension is distance, second dimension is theta
lambda(1) = 457.9e-9;
lambda(2) = 476.5e-9;
lambda(3) = 496.5e-9;
lambda(4) = 501.7e-9;
lambda(5) = 528.7e-9;
lambda(6) = 658e-9;
j = 1;
   [ I, P ] = OneSlitDiffraction(0.75e-5,1.76e-2,44.1e-6,lambda(j));
%    plot(P,I,P+13,W_L2(:,j)*.2);
plot(P,I)
   ylim([0,0.07])
%%   
j = 2;
   [ I, P ] = OneSlitDiffraction(0.75e-5,1.76e-2,44.1e-6,lambda(j));
   plot(P,I,P+13,W_L2(:,j)*0.15);
   ylim([0,0.07])
   
%%
j = 3;
   [ I, P ] = OneSlitDiffraction(0.75e-5,1.76e-2,44.1e-6,lambda(j));
   plot(P,I,P+13,W_L2(:,j)*0.57);
   ylim([0,0.07])
%%
j = 4;
   [ I, P ] = OneSlitDiffraction(0.75e-5,1.76e-2,44.1e-6,lambda(j));
   plot(P,I,P+13,W_L2(:,j)*0.35);
   ylim([0,0.07])
   
%%
j = 5;
%    [ I, P ] = OneSlitDiffraction(0.75e-5,1.76e-2,44.1e-6,lambda(j));
[ I, P ] = OneSlitDiffraction(0.77e-5,1.76e-2,44.1e-6,528.7e-9);
   plot(P,I,P+12,W_L2(:,j)*0.26);
   ylim([0,0.07])   
%%   
j = 1;
   [ I, P ] = OneSlitDiffraction(0.75e-5,3.62e-2,44.1e-6,lambda(j));
   plot(P,I,P+4,W_L20(:,j)*0.3);
   ylim([0,0.07])  
   
%%   
j = 2;
   [ I, P ] = OneSlitDiffraction(0.75e-5,3.62e-2,44.1e-6,lambda(j));
   plot(P,I,P+4,W_L20(:,j)*0.3);
   ylim([0,0.07])  
%%   
j = 3;
   [ I, P ] = OneSlitDiffraction(0.75e-5,3.62e-2,44.1e-6,lambda(j));
   plot(P,I,P+4,W_L20(:,j)*0.7);
   ylim([0,0.07]) 
%%   
j = 4;
   [ I, P ] = OneSlitDiffraction(0.75e-5,3.62e-2,44.1e-6,lambda(j));
   plot(P,I,P+4,W_L20(:,j)*0.54);
   ylim([0,0.07]) 
%%   
j = 5;
   [ I, P ] = OneSlitDiffraction(0.75e-5,3.62e-2,44.1e-6,lambda(j));
   plot(P,I,P+4,W_L20(:,j)*0.84);
   ylim([0,0.07])      
   
%%  ---- Dif on slit: The different way to find pixel size
j = 4;
   plot(P+13,W_L2(:,j),P+4,W_L20(:,j));
%    ylim([0,0.07]) 
%% Pixel size determination - shifting kamera
Theta = -3:.01:3; % The angle range
X  = 44.1e-6*pi*sind(Theta)./lambda(j);
I = (sin(X)./X).^2;  % The angle destribution of intensity
plot(Theta,I)
ylim([0 0.06])
Ld = 18; % shifted distance [mm]
Pl = 40; % shifted boundary [Pix]
Psize = (18*tand(0.93)/40)*1e3; % Pixel size [um]
%% comparison between near-axes and full theory
j = 10;
plot(180*(pi/2+Theta)/pi,I(j,:),180*(pi/2+Theta_NA(end:-1:1))/pi,51*I_NA(j,:))
