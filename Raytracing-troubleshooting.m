%% Ray tracing: troubleshooting
clear;
% Frames from the experimental movie. 
% The current frame and sum of frames.
load('Frames_Sf_Cf.mat'); 
% Creation of the electrode borders
% The center of electrode coincide with origin of coordinate system
%% Numerical experiment. We need to generate theoretical image with
%   distortion and try to reconstruct it.
% Generation  of pattern: The IM_Mie was generated by PikeReader for file;
%  CCD position = [ 83.5, 5, -44 ]; There was a mistake. When we generate
%  ray tracing we need only distance between CCD and center of the trap. Shift is not provided in theory.
cd('D:\!Work\!Projects\Aberration correction');
 clear;
 load('RayTrasing_Mie.mat')
 Te = linspace(Theta(1),Theta(end),640);
  plot(Te*180/pi,sum(IM_Mie,1),Theta*180/pi,It,'linewidth',2); grid on;
xlabel('\theta','fontsize',16);
ylabel('I [arb.u.]','fontsize',14)
set(gca,'fontsize',14);
legend('Reconstructed','Original','fontsize',14)
%% After ray trasing. No shifting !!!
% The angles are correct but intensity is not correct.
figure;
plot(180*(ThetaCor2+pi/2)/pi,1.2e2*Icorected(1,:),...
      Theta*180/pi,It,'linewidth',2);
grid on;
xlabel('\theta','fontsize',16);
ylabel('I [arb.u.]','fontsize',14)
set(gca,'fontsize',14);
legend('Reconstructed','Original','fontsize',14)

%% Intesity desribution correction. From PikeReader
% We need to calculate the intensity distribution across the image, and
% divide experimental image by this theoretically calculation.
% Here, we have the corrected image!
plot( 180*(ThetaC+pi/2)/pi,...
      IC,180*Theta/pi,It,...
      'linewidth',2); 
grid on;
xlabel('\theta','fontsize',16);
ylabel('I [arb.u.]','fontsize',14)
set(gca,'fontsize',14);
legend('Reconstructed','Original','fontsize',14)  
%% IM - Image was created by using sine wave
imtool(IM);
%% IM_cor - Image represents the intensity distribution after lens system.
%  It is used for intensity correction of experimental data
imtool(IM_Cor)
%% IM_Mie - Image was created by using Mie theory
imtool(IM_Mie)
%% IM_R - Image was created by using R dependence of intensity
imtool(IM_R)
%% Drawing the data with and without intensity correction
cd('D:\!Work\!Projects\Aberration correction');
 clear;
 load('I_Theta_3_AfterInt_CorrectionResults.mat')
n = 13;
plot(Theta,I(n,:)./max(I(n,:)),Theta,I_Cor(n,:)./max(I_Cor(n,:)))
%% Radius estimation for corrected data
% Mie pattern generation
clear It err scale rf
r = 8.8e3:-1:8.45e3;
   m = 1.447;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
It = GeneratePattern(r,m,(Theta+pi/2),waves);

[err scale] = ReferenceDistance(I_Cor,It);
[err1 scale1] = ReferenceDistance(I,It);
%
for i = 1:(size(err,1))
    [C id] = min(err(i,:));
    rf(1,i)=r(id);
    [C id1] = min(err1(i,:));
    rf(2,i)=r(id1);
%     plot(Theta,I( ((i*10)+1),:),Theta,It(id,:)*scale(((i*10)+1),id),...
%          Theta,I_Cor( ((i*10)+1),:),Theta,It(id1,:)*scale1(((i*10)+1),id1 ) );

%     pause(0.3);
end
plot(rf(1,:),'.');hold on;
plot(rf(2,:),'p','color','r');hold off;
legend('Corrected data','Not corrected data')
xlabel('N [ numbers ]','fontsize',14);
ylabel('R [nm]','fontsize',14);
set(gca,'fontsize',14)
%% Radius estimation: Decreasing the radius step
% Mie pattern generation
clear It err scale
r = 8.8e3:-.005:8.45e3;
   m = 1.447;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
It = GeneratePattern(r,m,(Theta+pi/2),waves);
%% Because we have "out of memory" problem we need to run these calculations in loop
rf(3,:) = zeros(1,length(rf(1,:)));
wb = waitbar(0,'Distance calculations');
for ii = 1:size(I_Cor,1)
    waitbar(ii/size(I_Cor,1),wb);
[err scale] = ReferenceDistance(I_Cor(ii,:),It);
    [C id] = min(err);
rf(3,ii) = r(id);
end
close(wb)
figure;
plot(rf(1,:),'b'); hold on;
plot(rf(2,:),'g');
plot(rf(3,:),'r'); hold off;

%% The same for red channel
cd('D:\!Work\!Projects\Aberration correction');
 clear;
%  load('I_Theta_red.mat')
 load('I_Theta_red_Corrected.mat')
%% 
clear It err scale
r = 8.8e3:-.01:8.45e3;
   m = 1.447;
   waves.wavelength   = 658;
   waves.theta        = 0; 
   waves.polarization = 1;
It = GeneratePattern(r,m,(Theta+pi/2),waves);


[err scale] = ReferenceDistance(I_Cor(1:10:end,end:-1:1),It);
%% ploting data
for i = 1:(size(err,1))
    [C id] = min(err(i,:));
    rf(1,i) = r(id);
%     
%     plot(Theta,I( ((i*10)+1),:),Theta,It(id,:)*scale(((i*10)+1),id),...
%          Theta,I_Cor( ((i*10)+1),:),Theta,It(id1,:)*scale1(((i*10)+1),id1 ) );
% 
%     pause(0.3);
% 
    plot(Theta,I_Cor( i,end:-1:1),Theta,It(id,:)*scale(i,id));
    pause(0.1);

end
plot(rf,'.');
legend('Corrected data','Not corrected data')
xlabel('N [ numbers ]','fontsize',14);
ylabel('R [nm]','fontsize',14);
set(gca,'fontsize',14)
%% For Jastice
RF = fit((1:2501).',R(:,2),'smooth');
plot(1:2501,RF(1:2501)); hold on;
vr = linspace(1,2501,3e4);
plot(vr,RF(vr),'r'); hold off;
%%
   m = 1.447;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
It = GeneratePattern(RF(vr)*1e9,m,(Theta+pi/2),waves);
%%
plot(sum(It,2))
%% --> 17/12/2014 -> Jastice's data Test
%% ====== Drawing the data with and without intensity correction
cd('D:\!Work\!Projects\Aberration correction');
 
n = 100;
plot(180*(Theta+pi/2)/pi,I(n,:)./max(I(n,:)),180*(Theta+pi/2)/pi,I_Cor(n,:)./max(I_Cor(n,:)))
%% ======= Find parameters for single frame
clear It err scale rf err scale
r = 10e3:-.5:9.35e3;
   m = 1.45;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
% ====== May be, I make  mistake, in angle calculation
% New angle, recalculation
Deg_Theta = 180*Theta(end)/pi;
Range_Theta = [-17.05,16.05];  % in degree
% ======= Rescaled theta
 S_Theta = linspace(pi*Range_Theta(1)/180,pi*Range_Theta(2)/180,size(I_Cor,2));  
It = GeneratePattern(r,m,(S_Theta+pi/2),waves);
Ve = 1; %1:100:size(I_Cor,1);
[err scale] = ReferenceDistance(I_Cor(Ve,:)-min(I_Cor(Ve,:)),It);

    [C id] = min(err);
    rf(1,i) = r(id);
%%    
    plot( Theta+pi/2,(I_Cor( Ve,:)-min(I_Cor( Ve,:))),...
          S_Theta+pi/2, It(id,:)*scale(id)*1.9 );    grid on;
legend('Exp. data','Theory')


%% ======== Radius estimation For Test avi
clear It err scale rf err scale

r = 12.0e3:-.5:1.45e3;
   m = 1.45;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
   Deg_Theta = 180*Theta(end)/pi;
Range_Theta = [-17.01,16.01];  % in degree
 S_Theta = linspace(pi*Range_Theta(1)/180,pi*Range_Theta(2)/180,size(I_Cor,2));  
It = GeneratePattern(r,m,(S_Theta+pi/2),waves);
Ve = 1:1:size(I_Cor,1);
[err scale] = ReferenceDistance(I_Cor(Ve,:),It);
% ploting data
for i = 1:(size(err,1))
    [C id] = min(err(i,:));
    rf(1,i) = r(id);
%     plot(S_Theta,I_Cor( Ve(i),:),S_Theta,It(id,:)*scale(i,id));grid on;
%     legend('Exp. Data','Theory');
%     pause(0.2);

end
figure;
plot(rf,'.');
legend('Corrected data','Not corrected data')
xlabel('N [ numbers ]','fontsize',14);
ylabel('R [nm]','fontsize',14);
set(gca,'fontsize',14)


%% ======== Out of memory problem Radius estimation For Test avi
clear It err scale rf err scale
r = 12.0e3:-.5:1.45e3;
   m = 1.45;
   waves.wavelength   = 458;
   waves.theta        = 0; 
   waves.polarization = 0;
   Deg_Theta = 180*Theta(end)/pi;
Range_Theta = [-17.01,16.01];  % in degree
 S_Theta = linspace(pi*Range_Theta(1)/180,pi*Range_Theta(2)/180,size(I_Cor,2));  
It = GeneratePattern(r,m,(S_Theta+pi/2),waves);

rf = zeros(1,size(I_Cor,1));
    wb = waitbar(0,'Distance calculations');
    N = size(I_Cor,1);
    SN = num2str(N);
        for ii = 1:N
            waitbar(ii/size(I_Cor,1),wb,['Curent point -> ' num2str(ii),'Number ', SN ]);
            [err scale] = ReferenceDistance(I_Cor(ii,:)-min(I_Cor(ii,:)),It);
            [C id] = min(err);
            rf(ii) = r(id);
        end
close(wb)
figure;
plot(rf,'.');grid on;
%% ===== Working with recived data
plot(rf,'.')
