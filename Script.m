%% Do testowania GPU
clear;
load('GPU_Tests_Workspace.mat')
DrawLensSystem(handles.S);
title('Old version')
hold on;

Dw = linspace(-handles.S.dW/2,handles.S.dW/2,10);
Dh = [-1 -.5 0 .5 1];
Pd = [ handles.S.ld, 0, 0 ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,handles.S);
         plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
Yout = zeros(5,length(Dw));
for j = 1:length(Dh)
for i = 1 : length(Dw)
        Pd = [ handles.S.ld, Dw(i), Dh(j) ]; % Points on the diaphragm plane 
        P = RayTracing(Pd,handles.S);
        Yout(j,i)=P(7,2);
        plot3(P(:,1),P(:,2),P(:,3));
        plot3(P(:,1),P(:,2),P(:,3),'*','color','r');
   
end
end
hold off;
% view([1,0,0])

% GPU Part
figure;
DrawLensSystem(handles.S);
title('GPU calculation');
hold on;

Dw = single(linspace(-handles.S.dW/2,handles.S.dW/2,10));
Dh = single([-1 -.5 0 .5 1]);
Pl = single(ones(1,10).*handles.S.ld);
handles.shX = single(handles.shX);
handles.shY = single(handles.shY);
handles.S.m2 = single(handles.S.m2); 
         %[P,IM]=RayTracingCUDA(Pl,Dw,Dh,handles);
         P=RayTracingCUDA(Pl,Dw,Dh,handles);
         %disp(P);

Yout = zeros(5,length(Dw));
for j = 1:length(Dh)
for i = 1 : length(Dw)
        plot3(P(1,:,j,i),P(2,:,j,i),P(3,:,j,i));
        plot3(P(1,:,j,i),P(2,:,j,i),P(3,:,j,i),'*','color','r');
   
end
end
hold off;
% view([1,0,0])
