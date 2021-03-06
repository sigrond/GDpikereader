function DrawLensSystem(S)
% This function draws  the optical system.
% S -  the structure with parameters of the optical system generated by
% SetSystem.m function
% clf
% Calculation of sphere center position
S.Cs1  = S.l1 - S.R(1) + S.g; 
S.Cs2  = S.Cs1 + S.ll + 2*S.R(2);
%  The axes of the  coordinate system
plot3([-10 S.lCCD ],[0 0],[0 0],...
      [0 0],[-S.D/2 S.D/2],[0 0],...
      [0 0],[0 0],[-S.D/2 S.D/2],'linewidth',2,'linestyle','--' );  
  hold on; grid on;
  xlabel 'X'; ylabel 'Y'; zlabel 'Z';
%---------------------
% There we draw the first diaphragm
  df = ones(100,300)*S.ld;
  B = 1.5;   % Cover of diaphragm (width of diaphragm's border)
  vx = linspace(-S.dW/2-B,S.dW/2+B,300);  
  vy = linspace(-S.dH/2-B,S.dH/2+B,100);
% search for element's indexes which placed inside diaphragm
  ix = find( (vx >= -S.dW/2 ) & ( vx <= S.dW/2 ) );
  iy = find( (vy >= -S.dH/2 ) & ( vy <= S.dH/2 ) );
  mx = meshgrid(vx,vy);
  my = meshgrid(vy,vx)';
% It will be hole in diaphragm  
  mx(iy,ix) = NaN; 
  my(iy,ix) = NaN;
  
  surf(df,mx,my,df,'linestyle','none')
%------------------------  
% There we draw the second diaphragm
  df2 = ones(100,300)*S.ld2;
  mx = meshgrid(vx,vy);
  my = meshgrid(vy,vx)';
%  Making a hole 
for ki = 1:length(vx)
    for kj = 1:length(vy)
        if norm( [ vx(ki) vy(kj) ] ) <= S.RDph
            df2(kj,ki) = NaN;
        end
    end
end
  surf(df2, mx, my, df, 'linestyle', 'none');
  surf(df2 + S.W2, mx, my, df, 'linestyle','none');  
%---------------------  
% Drawing the lens  
[z,y,x] = cylinder(S.D/2,50);
l = sqrt(S.R(1)^2-(S.D/2)^2);            %  S.Cs1 + l - is position of the second surface
plot3(x(1,:) + S.Cs1 + S.R(1) - S.g  , y(1,:),z(1,:),...       %left to right; first surface
      x(1,:) + S.Cs1 + l, y(1,:), z(1,:),'linewidth',2 );  %               second surface 
% Drawing the spherical surface of lens
theta =   atand(S.D/2/l);             % Angular diameter of the lens
vt = 90 +linspace(-theta,theta,50);   % vector of angles 
x = S.Cs1 + sind(vt) * S.R(1);
y =  zeros(size(vt));
z =  cosd(vt) * S.R(1);
x = [S.Cs1 + S.R(1) - S.g x S.Cs1 + S.R(1) - S.g]; % adding extreme points
y = [y(1) y y(end) ];
z = [z(1) z z(end) ];
plot3( x,y,z,x,z,y,'color','b','linewidth',2);
%--------------------------------                                 
% Drawing the second lens
[z,y,x] = cylinder(S.D/2,50);
plot3(x(1,:) + S.Cs2 - S.R(2) + S.g  , y(1,:),z(1,:),...       % righ to tleft ; first surface
      x(1,:) + S.Cs2 - l, y(1,:), z(1,:),'linewidth',2 );  %               second surface 
% Drawing the spherical surface of lens
theta =   atand(S.D/2/l);
vt = 270+linspace(-theta,theta,50);
x = S.Cs2 + sind(vt) * S.R(2);
y =  zeros(size(vt));
z =  cosd(vt) * S.R(2);
x = [S.Cs2 - S.R(2) + S.g x S.Cs2 - S.R(2) + S.g];
y = [y(1) y y(end) ];
z = [z(1) z z(end) ];
plot3( x,y,z,x,z,y,'color','b','linewidth',2);
% ----------------- 
%  Drawing the diaphragms and CCD detector
  df = ones(48,64)*S.lCCD;
  vx = linspace(-S.CCDW/2,S.CCDW/2,64);
  vy = linspace(-S.CCDH/2,S.CCDH/2,48);
  mx = meshgrid(vx,vy);
  my = meshgrid(vy,vx)';
  surf(df,mx,my,df,'linestyle','none')
                                     hold off;
                                 view([0,0,1])