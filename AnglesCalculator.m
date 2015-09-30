function h = AnglesCalculator(h)
% This function calculates correct angles distribution for the image with
% aberration.
% 
% Finding pixels inside the region of interest
%
[r,c] = find( h.Bw );
% Recalculation pixels to meters and shifting image at the origin
Y = ( ( c - h.S.CCDPW/2 - h.shX) * h.S.PixSize );
Z = ( ( r - h.S.CCDPH/2 - h.shY) * h.S.PixSize );
% Number of points for recalculation
N = length(Y);
% Coordinates of points before and after lens system respectively
h.M_in = zeros(length(Y),2);
h.M_out = zeros(length(Z),2);
h.ThetaPhiR = zeros(length(Z),3);
% Starting point
Z0 = 0;
% Time estimation
i = 1;
tic;
YF = Y(i);
    ZF = Z(i);
    
    h.Err = Inf;         % exit condition
    epsilon = 1e-6;      % precision
    count = 1;           % number of iteration
    % Coordinate descent method
    while ( count < 4 )
        count = count + 1;
        Y0 = fminbnd( @FdY,-h.S.dW/2, h.S.dW/2 );
        Z0 = fminbnd( @FdZ,-h.S.dH/2, h.S.dH/2 );
        
    end
tt = toc;

% End time estimation
Str = sprintf('Angles calculation. First channel. \n Estimated time -> %2.1f [min]',tt*N/60);
wb = waitbar(0,Str);

for i = 1:N
    waitbar(i/N,wb);
    
    % Points that we are looking for
    YF = Y(i);
    ZF = Z(i);
    
    h.Err = Inf;         % exit condition
    epsilon = 1e-6;   % precision
    count = 1;             % number of iteration
    % Coordinate descent method
    while ( h.Err > epsilon )&&( count < 4 )
        count= count + 1;
        Y0 = fminbnd( @FdY,-h.S.dW/2, h.S.dW/2 );
        Z0 = fminbnd( @FdZ,-h.S.dH/2, h.S.dH/2 );
        
    end
    P2 = [h.S.ld,Y0,Z0];
    R = RayTracing( P2,h.S );
    
    h.M_in(i,1) = Y0;
    h.M_in(i,2) = Z0;
    h.M_out(i,1) = R(7,2);
    h.M_out(i,2) = R(7,3);
    
    h.ThetaPhiR(i,1) = atan( Y0/( h.S.ld - R(1,1) ) );           % Theta
    h.ThetaPhiR(i,2) = atan( Z0/( h.S.ld - R(1,1) ) );           % Phi
    h.ThetaPhiR(i,3) = norm( [ ( h.S.ld - R(1,1) ), Y0, Z0 ] );  % Distance 
end
save('h.mat');
close(wb);
h.r = r;
h.c = c;
% checking solutions
figure('name','Checking solutions');
    subplot(1,2,1); 
    plot(Y,Z,'p'); grid on;
    xlabel('Y[mm]');ylabel('Z[mm]')
    hold on;
    plot(h.M_out(:,1),h.M_out(:,2),'.','color','r');
    xlabel('Y[mm]');ylabel('Z[mm]')
    hold off
    legend('Experimental points','Found points');
    subplot(1,2,2);
    plot(h.M_in(:,1),h.M_in(:,2),'.' );grid on;
    xlabel('Y[mm]');ylabel('Z[mm]');
    legend('Points before lens system');
%% ==== Auxiliary functions =====
    function error = FdY( Y0 )
        %          Pd = [ S.ld, H.vW(i), H.vH(i) ];
        
        P2 = [h.S.ld,Y0,Z0];
        R = RayTracing( P2,h.S );
        error = abs( R( 7, 2 ) - YF ) + abs( R( 7, 3 ) - ZF );
        h.Err = error;
        
    end
    function error = FdZ(Z0 )
        %          Pd = [ S.ld, H.vW(i), H.vH(i) ];
        
        P2 = [h.S.ld,Y0,Z0];
        R = RayTracing( P2,h.S );
        error = abs( R( 7, 2 ) - YF ) + abs( R( 7, 3 ) - ZF );
        h.Err = error;
        
    end

end
