function varargout = PikeReader(varargin)
%  
% 

% Last Modified by GUIDE v2.5 29-Sep-2015 15:01:44

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PikeReader_OpeningFcn, ...
                   'gui_OutputFcn',  @PikeReader_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before PikeReader is made visible.
function PikeReader_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PikeReader (see VARARGIN)

% Choose default command line output for PikeReader
handles.output = hObject;
handles.fn = [];
handles.sF = [];
handles.hl = [];
handles.hImSh = []; % handles to imshow object
Frame = FrameRider(hObject,handles);
handles.cF = Frame;
handles.S = SetSystem;    
handles.TshX = 0;
handles.shX  = 0;
handles.TshY = 0;
handles.shY  = 0;
handles.LW   = 0;
handles.shLW = 0;
% handles for masks
handles.BWR = [];
handles.BWG = [];
handles.BWB = [];

set(handles.edAperture,'string',num2str(handles.S.D));
set(handles.edPdrop,'string',num2str(handles.S.Pk));
set(handles.edCCD,'string',[num2str(handles.S.lCCD),', 0',', 0']);
set( handles.ed_Sh_l1,'string', num2str( handles.S.l1 ) );
Draw(hObject,handles);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes PikeReader wait for user response (see UIRESUME)
% uiwait(handles.Fig1);


% --- Outputs from this function are returned to the command line.
function varargout = PikeReader_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
%% My functions
%=============== My functions =========================================
function Frame = FrameRider(hObject,handles)
% reads frame from movie ->
if isempty( handles.fn )
%     In case when movie is not loaded
     Frame = zeros(480,640,3);   
else
     mov = AviReadPike_Split( handles.fn,handles.nom );
     Frame =  squeeze( mov );
end
% ---------------------------------------------------------------------
function Draw(hObject,handles)
 
        %  draws image
    if get(handles.chSumFrames,'value')
        % get sum frames
        if ~isempty(handles.sF)
            temp = handles.sF;
        else
            % if there isn't sum of frame, then draws current frame instead
            h = errordlg('Sum of Frames not found','Sum Error'); uiwait(h);
            temp = handles.cF;
            set(handles.chSumFrames,'value',0)
        end
    else
        % get current frame
        temp = handles.cF;
    end
    Frame =  zeros(size(temp));
    % Choosing the channel of observation 
    if get(handles.chR,'value')
        Frame(:,:,1) = temp(:,:,1);
    end
    if get(handles.chG,'value')
        Frame(:,:,2) = temp(:,:,2);
    end
    if get(handles.chB,'value')
        Frame(:,:,3) = temp(:,:,3);
    end
    % Correction of the level of intensity
    if get(handles.chAdjust,'value')
        Frame = Frame.*str2double( get(handles.edAdjust,'string'));
    end
    % If there is no path, then show the empty frame
    if isempty( handles.hImSh )
        if isempty( handles.fn )
           handles.hImSh = imshow( handles.cF);
        else
           handles.hImSh = imshow(uint16(Frame));
        end
        if isempty(handles.hl)
            hold on;
            % first channel
            handles.hl(1) = plot( [0, 0],[0, 0],'color',[0.9,0.9,0.9] ); % The red channel
            % additional vertical lines
            handles.hl(2) = plot( [0, 0],[0, 0],'color',[0.9,0.9,0.9] );
            handles.hl(3) = plot( [0, 0],[0, 0],'color',[0.9,0.9,0.9] );
            % second channel
            handles.hl(4) = plot( [0, 0],[0, 0],'color','r' );
            handles.hl(5) = plot( [0, 0],[0, 0],'color','r' );
            handles.hl(6) = plot( [0, 0],[0, 0],'color','r' );
            % third channel
            handles.hl(7) = plot( [0, 0],[0, 0],'color','g' );
            handles.hl(8) = plot( [0, 0],[0, 0],'color','g' );
            handles.hl(9) = plot( [0, 0],[0, 0],'color','g' );
            hold off;
            guidata(hObject, handles);
        end
    else
        if isempty( handles.fn )
           set(handles.hImSh,'Cdata',handles.cF);
        else
           set(handles.hImSh,'Cdata',uint16(Frame));
        end
    end
    % If sight checkbox is active then draw theoretical image
    if get(handles.chSight,'value')
       
        % setting the wavelength value
        if get(handles.chR,'value')
            handles.S.lambda = str2double(get(handles.edR,'string'));
            handles.S.m2 = Calculate_m(25,handles.S.lambda, 'BK7');
            handles.ChKey = 1;
            guidata(hObject,handles);
                Sight(hObject,handles);
        else
            set(handles.hl(1),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(2),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(3),'xdata',[0,0],'ydata',[0,0]);
        end
        if get(handles.chG,'value')
            handles.S.lambda = str2double(get(handles.edG,'string'));
            handles.S.m2 = Calculate_m(25,handles.S.lambda, 'BK7');
             handles.ChKey = 2;
             guidata(hObject,handles);
                Sight(hObject,handles);
               
        else
            set(handles.hl(4),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(5),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(6),'xdata',[0,0],'ydata',[0,0]);
        end
        if get(handles.chB,'value')
            handles.S.lambda = str2double(get(handles.edB,'string'));
             handles.S.m2 = Calculate_m(25,handles.S.lambda, 'BK7');
             handles.ChKey = 3;
            guidata(hObject,handles);
                Sight(hObject,handles);
        else
            set(handles.hl(7),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(8),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(9),'xdata',[0,0],'ydata',[0,0]);
        end
        guidata(hObject,handles)
    end
% ---------------------------------------------------------------------
function Sight(hObject,handles)
 
 % setting the droplet's position
 handles.S.Pk = str2num(get(handles.edPdrop,'string'));
 % setting the CCD's position
 vel = str2num(get(handles.edCCD,'string'));
  handles.S.lCCD = vel(1);
  handles.shX = vel(2);
  handles.shY = vel(3);
  % setting the effective aperture
  handles.S.efD = str2double(get(handles.edAperture,'string'));
  % setting the position of additional line
  handles.shLW = str2double(get(handles.edLineSh,'string'));
  handles.S.CCDH = handles.S.CCDPH * handles.S.PixSize;  % height of CCD
  handles.S.CCDW = handles.S.CCDPW * handles.S.PixSize;  % width  of CCD
  % setting the distance between center of the trap and  first lens
  handles.S.l1   = str2double( get( handles.ed_Sh_l1,'string' ) );
guidata(hObject,handles);
DrawTheorImage(hObject,handles);
%-------------------------------------------------------------------------
function Br = BorderCreation(hObject,handles)
% This function creates border points. The position of border points depends on geometry
% of trap and droplet's position.
% Trap center coincides with the center of
% the coordinate system
%
Br = zeros(4*handles.S.N,3);       % vector of border points
% calculation of position for the 4 outer points
alpha = asin(handles.S.dW/2/handles.S.R_midl_El);
[X(1),Y(1)] = pol2cart(alpha,handles.S.R_midl_El);
[X(2),Y(2)] = pol2cart(-alpha,handles.S.R_midl_El);
Z(1) = -handles.S.dH/2;
Z(2) = handles.S.dH/2;
P(1,:) = [X(1),Y(1),Z(1)];
P(2,:) = [X(1),Y(1),Z(2)];
P(3,:) = [X(2),Y(2),Z(2)];
P(4,:) = [X(2),Y(2),Z(1)];

V = P(4,:) - handles.S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*handles.S.Pk(1)+V(2)*handles.S.Pk(2));
C = handles.S.Pk(1)^2 + handles.S.Pk(2)^2 - (handles.S.R_dis_Ring)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(5,:) = [ handles.S.Pk(1)+V(1)*t handles.S.Pk(2)+V(2)*t -handles.S.dH/2];

V = P(1,:) - handles.S.Pk; % left (or right) border
V = V/norm(V);
A = V(1)^2 + V(2)^2;
B = 2*(V(1)*handles.S.Pk(1)+V(2)*handles.S.Pk(2));
C = handles.S.Pk(1)^2 + handles.S.Pk(2)^2 - (handles.S.R_dis_Ring)^2;
D = B^2-4*A*C;
t = (-B+sqrt(D) )/2/A;
P(6,:) = [ handles.S.Pk(1)+V(1)*t handles.S.Pk(2)+V(2)*t -handles.S.dH/2];

P(7,1:2) = P(6,1:2);
P(7,3)   = handles.S.dH/2;

P(8,1:2) = P(5,1:2);
P(8,3)   = handles.S.dH/2;
%
Br(1:handles.S.N,1) = ones(1,handles.S.N)*P(1,1);
Br(1:handles.S.N,2) = ones(1,handles.S.N)*P(1,2);
Br(1:handles.S.N,3) = linspace(P(2,3),P(1,3),handles.S.N);
V1 = P(6,:); % directing vector from origin to point 6
V2 = P(5,:); % directing vector from origin to point 5
Bt1 = acos( dot( V1(1:2) ,[1,0])/norm( V1(1:2) ));
Bt2 = acos( dot(V2(1:2),[1,0])/norm(V2(1:2)));
VBt = linspace(Bt1,-Bt2,handles.S.N);
%
[X,Y] = pol2cart(VBt,handles.S.R_dis_Ring);

Br((handles.S.N+1):2*handles.S.N,1) = X;
Br((handles.S.N+1):2*handles.S.N,2) = Y;
Br((handles.S.N+1):2*handles.S.N,3) = -ones(1,handles.S.N)*handles.S.dH/2;

Br((2*handles.S.N+1):3*handles.S.N,1) = ones(1,handles.S.N)*P(4,1);
Br((2*handles.S.N+1):3*handles.S.N,2) = ones(1,handles.S.N)*P(4,2);
Br((2*handles.S.N+1):3*handles.S.N,3) = linspace(P(4,3),P(3,3),handles.S.N);

VBt = linspace(-Bt2,Bt1,handles.S.N);
%
[X,Y] = pol2cart(VBt,handles.S.R_dis_Ring);
Br((3*handles.S.N+1):4*handles.S.N,1) = X;
Br((3*handles.S.N+1):4*handles.S.N,2) = Y;
Br((3*handles.S.N+1):4*handles.S.N,3) = ones(1,handles.S.N)*handles.S.dH/2;
%-------------------------------------------------------------------------
function DrawTheorImage(hObject,handles)
% generation of border points
handles.Br = BorderCreation(hObject,handles);
% ray tracing calculation
for i = 1:size(handles.Br,1)
       Pd = [ handles.Br(i,1), handles.Br(i,2), handles.Br(i,3) ]; % Points on the diaphragm plane 
       P = RayTracing(Pd,handles.S);
             if size(P,1) == 7
                 W1(i) = P(end,2);
                 Hi1(i) = P(end,3);
             else 
              % Terminate the rays that don't hit the CCD element   
                 W1(i) = NaN;
                 Hi1(i) = NaN;
             end
end
% Recalculation meters to pixels
% shifting the  origin to middle of the image.
% The center of image isn't placed  on [0,0] point, but on [240,320] point
 handles.R1(1,:) = (handles.S.CCDW/2 + W1)/handles.S.PixSize;  % [ Pix ]
 handles.R1(2,:) = (handles.S.CCDH/2 + Hi1)/handles.S.PixSize; % [ Pix ]
  
% drawing the additional symetrical lines
handles.N = 20;
handles.LH = linspace(-handles.S.dH/2,handles.S.dH/2,handles.N);
handles.L = zeros(2,handles.N);

for ii = 1:handles.N
 Pd = [ handles.S.ld, handles.LW + handles.shLW, handles.LH(ii) ];
         P1 = RayTracing(Pd,handles.S);
         handles.L(1,ii)= (handles.S.CCDW/2 + P1(7,2))/handles.S.PixSize;
         handles.L(2,ii)= (handles.S.CCDH/2 + P1(7,3))/handles.S.PixSize;
         
 Pd = [ handles.S.ld, handles.LW - handles.shLW, handles.LH(ii) ];
         P1 = RayTracing(Pd,handles.S);
         handles.L(3,ii)= (handles.S.CCDW/2 + P1(7,2))/handles.S.PixSize;
         handles.L(4,ii)= (handles.S.CCDH/2 + P1(7,3))/handles.S.PixSize;        
end
switch handles.ChKey
    case 1 % Red channel
         set(handles.hl(1),'xdata',handles.R1(1,:) +  handles.shX,...
                           'ydata',handles.R1(2,:) + handles.shY );
         set(handles.hl(2),'xdata',handles.L(1,:) + handles.shX,...
                           'ydata',handles.L(2,:)+ handles.shY);
         set(handles.hl(3),'xdata',handles.L(3,:) + handles.shX,...
                           'ydata',handles.L(4,:)+ handles.shY);
    case 2
         set(handles.hl(4),'xdata',handles.R1(1,:) +  handles.shX,...
                           'ydata',handles.R1(2,:) + handles.shY );
         set(handles.hl(5),'xdata',handles.L(1,:) + handles.shX,...
                           'ydata',handles.L(2,:)+ handles.shY);
         set(handles.hl(6),'xdata',handles.L(3,:) + handles.shX,...
                           'ydata',handles.L(4,:)+ handles.shY);
    case 3
         set(handles.hl(7),'xdata',handles.R1(1,:) +  handles.shX,...
                           'ydata',handles.R1(2,:) + handles.shY );
         set(handles.hl(8),'xdata',handles.L(1,:) + handles.shX,...
                           'ydata',handles.L(2,:)+ handles.shY);
         set(handles.hl(9),'xdata',handles.L(3,:) + handles.shX,...
                           'ydata',handles.L(4,:)+ handles.shY);
    
end

guidata(hObject,handles)
%=============== End of  My functions =================================


% --- Executes on button press in pbLoad.
function pbLoad_Callback(hObject, eventdata, handles)
% pbLoad_Callback - wczytuje film *.avi
%I:\!From Justice\0.1ml\Splited
[handles.f,handles.dir] = uigetfile( {'*.avi';'*.*'},'Load files','D:\!Work\!Projects\Aberration correction\AVI','MultiSelect','on' );
% wy�wietlamy na panelu nazwe filmu
    if ischar( handles.f )
        set( handles.up1, 'title', handles.f );
        handles.avi_title = handles.f;
        handles.fn = [handles.dir handles.f];
        inf = aviinfo(handles.fn);
        handles.N_frames = inf.NumFrames; % total number of frames
        set(handles.slFrames,'max',handles.N_frames,'min',1,'value',1,...
            'sliderstep',[1/handles.N_frames 10/handles.N_frames],'enable','on');
        S = sprintf('Current frame: %d; Total number of frame %d',1, handles.N_frames);
        set(handles.upFrames,'Title',S);
        set(handles.pmPart,'string','1')
    elseif iscell( handles.f )
        s = sprintf('%s in %3.1f parts',handles.f{1},size(handles.f,2) );
        set( handles.up1,'title',s );

         handles.N_frames =0;
         for j = 1:length(handles.f)
             handles.fn = [handles.dir handles.f{j}];
             inf = aviinfo(handles.fn);
             handles.N_frames = handles.N_frames + inf.NumFrames; % total number of frames
         end
            handles.fn = [handles.dir handles.f{1}];
            inf = aviinfo(handles.fn);
            set(handles.slFrames,'max',inf.NumFrames,'min',1,'value',1,...
            'sliderstep',[1/inf.NumFrames 10/inf.NumFrames],'enable','on');
        S = sprintf('Current frame: %d; Total number of frame %d',1, handles.N_frames);
        set(handles.upFrames,'Title',S);
        set(handles.pmPart,'string',num2str( (1:length(handles.f))' ));
    elseif handles.f == 0
        set( handles.up1,'title','No files...' );
        set(handles.slFrames,'enable','off');
        S = sprintf('Current frame: %d; Total number of frame %d',0, 0);
        set(handles.upFrames,'Title',S);
        return
    end
handles.nom = 1;
Frame = FrameRider(hObject,handles);
handles.cF = Frame;
guidata(hObject,handles);
Draw(hObject,handles);
% --- Executes on button press in chR.
function chR_Callback(hObject, eventdata, handles)
% hObject    handle to chR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of chR



function edR_Callback(hObject, eventdata, handles)
% hObject    handle to edR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edR as text
%        str2double(get(hObject,'String')) returns contents of edR as a double


% --- Executes during object creation, after setting all properties.
function edR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in chG.
function chG_Callback(hObject, eventdata, handles)
% hObject    handle to chG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of chG



function edG_Callback(hObject, eventdata, handles)
% hObject    handle to edG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edG as text
%        str2double(get(hObject,'String')) returns contents of edG as a double


% --- Executes during object creation, after setting all properties.
function edG_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in chB.
function chB_Callback(hObject, eventdata, handles)
% hObject    handle to chB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of chB



function edB_Callback(hObject, eventdata, handles)
% hObject    handle to edB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edB as text
%        str2double(get(hObject,'String')) returns contents of edB as a double


% --- Executes during object creation, after setting all properties.
function edB_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in chAdjust.
function chAdjust_Callback(hObject, eventdata, handles)
% hObject    handle to chAdjust (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of chAdjust



function edAdjust_Callback(hObject, eventdata, handles)
% hObject    handle to edAdjust (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edAdjust as text
%        str2double(get(hObject,'String')) returns contents of edAdjust as a double


% --- Executes during object creation, after setting all properties.
function edAdjust_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edAdjust (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbSumFrames.
function pbSumFrames_Callback(hObject, eventdata, handles)
% this function calculates sum of frames
%
handles.nom = 1;
Frame = FrameRider(hObject,handles);
handles.sF = zeros( size( Frame ) );

count_step = str2double( get(handles.edSumFrameStep,'string') );
wb = waitbar(0,'Processing');
if iscell( handles.f )
    Nom = size( handles.f, 2 );
    for ii = 1 : Nom
        waitbar(ii/Nom,wb,['Processing segment number ' num2str(ii) ' from ' num2str(Nom)]);
        path = [handles.dir handles.f{ii}];
        inf = aviinfo( path );
        for j = 1:count_step:inf.NumFrames
            handles.nom = j;
            Frame = FrameRider(hObject,handles);
            handles.sF = handles.sF + Frame;
        end
    end
elseif ischar( handles.f )
    path = [ handles.dir handles.f ];
    inf = aviinfo( path );
    for j = 1:count_step:inf.NumFrames
        waitbar(j/inf.NumFrames,wb,'Processing...');        
        handles.nom = j;
        Frame = FrameRider(hObject,handles);
        handles.sF = handles.sF + Frame;
    end
end
close(wb);  
guidata( hObject, handles );
Draw( hObject, handles );


function edSumFrameStep_Callback(hObject, eventdata, handles)
% hObject    handle to edSumFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edSumFrameStep as text
%        str2double(get(hObject,'String')) returns contents of edSumFrameStep as a double


% --- Executes during object creation, after setting all properties.
function edSumFrameStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edSumFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in chSumFrames.
function chSumFrames_Callback(hObject, eventdata, handles)
% hObject    handle to chSumFrames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of chSumFrames


% --- Executes on button press in chSight.
function chSight_Callback(hObject, eventdata, handles)
% hObject    handle to chSight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Draw(hObject,handles);
if get(hObject,'value')~=1
      set(handles.hl(1),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(2),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(3),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(4),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(5),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(6),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(7),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(8),'xdata',[0,0],'ydata',[0,0]);
            set(handles.hl(9),'xdata',[0,0],'ydata',[0,0]);
end
    

% --- Executes on slider movement.
function slFrames_Callback(hObject, eventdata, handles)
% hObject    handle to slFrames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    handles.nom  = round(get(handles.slFrames,'value'));
    set(handles.upFrames,'title',num2str(handles.nom));
    Frame = FrameRider(hObject,handles);
    handles.cF = Frame;
    guidata(hObject, handles);
    Draw(hObject,handles);
    S = sprintf('Current frame: %d; Total number of frame %d',handles.nom, handles.N_frames);
        set(handles.upFrames,'Title',S);

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slFrames_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slFrames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on selection change in pmPart.
function pmPart_Callback(hObject, eventdata, handles)
% hObject    handle to pmPart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    vel = get(handles.pmPart,'value');
    handles.fn = [handles.dir handles.f{vel}];
    inf = aviinfo(handles.fn);
    set(handles.slFrames,'max',inf.NumFrames,'min',1,'value',1,...
        'sliderstep',[1/inf.NumFrames 10/inf.NumFrames]);
    handles.nom = 1;
    Frame = FrameRider(hObject,handles);
    handles.cF = Frame;
    Draw(hObject,handles);
    S = sprintf('Current frame: %d; Total number of frame %d',handles.nom, handles.N_frames);
    set(handles.upFrames,'Title',S);
    guidata(hObject,handles);
% Hints: contents = get(hObject,'String') returns pmPart contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pmPart


% --- Executes during object creation, after setting all properties.
function pmPart_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pmPart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in rbR.
function rbR_Callback(hObject, eventdata, handles)
% hObject    handle to rbR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.rbG,'value',0);
set(handles.rbB,'value',0);
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of rbR


% --- Executes on button press in rbG.
function rbG_Callback(hObject, eventdata, handles)
% hObject    handle to rbG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.rbB,'value',0);
set(handles.rbR,'value',0);
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of rbG


% --- Executes on button press in rbB.
function rbB_Callback(hObject, eventdata, handles)
% hObject    handle to rbB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.rbG,'value',0);
set(handles.rbR,'value',0);
Draw(hObject,handles);
% Hint: get(hObject,'Value') returns toggle state of rbB



function edPdrop_Callback(hObject, eventdata, handles)
% hObject    handle to edPdrop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edPdrop as text
%        str2double(get(hObject,'String')) returns contents of edPdrop as a double


% --- Executes during object creation, after setting all properties.
function edPdrop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edPdrop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edCCD_Callback(hObject, eventdata, handles)
% hObject    handle to edCCD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of edCCD as text
%        str2double(get(hObject,'String')) returns contents of edCCD as a double


% --- Executes during object creation, after setting all properties.
function edCCD_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edCCD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edAperture_Callback(hObject, eventdata, handles)
% hObject    handle to edAperture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 Draw(hObject,handles);

% Hints: get(hObject,'String') returns contents of edAperture as text
%        str2double(get(hObject,'String')) returns contents of edAperture as a double


% --- Executes during object creation, after setting all properties.
function edAperture_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edAperture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on mouse motion over figure - except title and menu.
function Fig1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to Fig1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% get(hObject,'CurrentPoint')


% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% get(hObject,'CurrentPoint')


% --- Executes on button press in pbAngleCalc.
function pbAngleCalc_Callback(hObject, eventdata, handles)
% This function calculates correct angles distribution for the image with
% aberration.
   handles = AnglesCalculator(handles);
   guidata(hObject,handles);


% --- Executes on button press in pbIntensCalc.
function pbIntensCalc_Callback(hObject, eventdata, handles)
% hObject    handle to pbIntensCalc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% TODO: Intensity correction!!!

% IC = evalin('base','IM_Cor');
% handles.NP = 700;
% Theta = linspace(min(handles.ThetaPhiR(:,1)),max(handles.ThetaPhiR(:,1)),handles.NP);
%  ip = find( handles.Bw ); % indexes of good points
%             Frame = IM./IC;
%       sf = smooth(handles.ThetaPhiR(:,1),  Frame(ip),50);
%       cf = fit(handles.ThetaPhiR(:,1), sf,'smooth');
%       I = cf(Theta);
% assignin('base','IC',I);
% assignin('base','ThetaC',Theta);

% END_TODO: Intensity correction!!!
IC = evalin('base','IM_Cor');
handles.NP = 700;
 FrameStep = str2num( get( handles.edFrameStep,'string') );
         handles.N_frames = 0;
         for j = 1:length(handles.f)
             handles.fn = [handles.dir handles.f{j}];
             inf = aviinfo(handles.fn);
             V = 1:FrameStep:inf.NumFrames;
             handles.N_frames = handles.N_frames + length(V); % total number of frames
         end



Theta = linspace( min( handles.ThetaPhiR(:,1) ), max( handles.ThetaPhiR(:,1) ), handles.NP );
I = zeros( handles.N_frames, handles.NP );
I_Corected = I;
nom = 1;
 ip = find( handles.Bw ); % indexes of good points
 wb = waitbar(0,'Angle Calculation');
 t = 0;

for i = 1 : length( handles.f ) % loop of  movie pieces 
    handles.fn = [handles.dir handles.f{i}];
    inf = aviinfo(handles.fn);
    for j = 1 : FrameStep: inf.NumFrames
        s = sprintf('Processing %d-th frame. \n Estimated time -> %f[min]; time left %f',...
                     j,handles.N_frames*t/60,(handles.N_frames-nom)*t/60);
        waitbar(i/length( handles.f ),wb,s);
        %  Choosing the channel
        handles.nom = j;
        handles.cF = FrameRider(hObject,handles);
        if get(handles.rbR,'value')
            Frame = handles.cF(:,:,1);
        end
        if get(handles.rbG,'value')
            Frame = handles.cF(:,:,2);
        end
        if get(handles.rbB,'value')
            Frame = handles.cF(:,:,3);
        end
       tic; 
       
      sf = smooth(handles.ThetaPhiR(:,1),  Frame(ip),50);
      cf = fit(handles.ThetaPhiR(:,1), sf,'smooth');
    
      I(nom,:) = cf(Theta);
      temp = Frame./IC;
      iNan = find(isnan(temp));
      temp(iNan) = 0;
      iNan = find(isinf(temp));
      temp(iNan) = 0;
      
      sfCor = smooth(handles.ThetaPhiR(:,1),temp(ip),50);
      cfCor = fit(handles.ThetaPhiR(:,1), sfCor,'smooth');
      I_Corected(nom,:) = cfCor(Theta);
      t = toc;
      nom = nom + 1;
    end
end
close(wb);
assignin('base','I',I);
assignin('base','I_Cor',I_Corected);
assignin('base','Theta',Theta)

%%
   guidata(hObject,handles);


function edLineSh_Callback(hObject, eventdata, handles)
% hObject    handle to edLineSh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles)
% Hints: get(hObject,'String') returns contents of edLineSh as text
%        str2double(get(hObject,'String')) returns contents of edLineSh as a double


% --- Executes during object creation, after setting all properties.
function edLineSh_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edLineSh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbLoadParam.
function pbLoadParam_Callback(hObject, eventdata, handles)
% hObject    handle to pbLoadParam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    Save = evalin('base','Save');
    handles.S = Save.S;  % load s structure;
    handles.Bw = Save.Bw;
% loading checkbox structure
    set(handles.rbR,'value',Save.rbR_value);
    set(handles.rbG,'value',Save.rbG_value);
    set(handles.rbB,'value',Save.rbB_value);
  %---
    set(handles.chR,'value',Save.chR_value);
    set(handles.chG,'value',Save.chG_value);
    set(handles.chB,'value',Save.chB_value);  
  %---
    set(handles.chSight,'value',Save.chSight);
    set(handles.chAdjust,'value',Save.chAdjust);
%     set(handles.chSumFrames,'value',Save.chSumFrames);
% Loading lambda edit boxes
    set(handles.edR,'string',Save.edR);
    set(handles.edG,'string',Save.edG);
    set(handles.edB,'string',Save.edB);
% Loading position paramiters
    set(handles.edPdrop,'string',Save.Pk);
    set(handles.edCCD,'string',Save.edCCD);
    set(handles.edAperture,'string',Save.edAperture);
    set(handles.edLineSh,'string',Save.shLW);
    set(handles.ed_Sh_l1,'string',Save.ed_Sh_l1);
% Loading frame step and Adjust box    
    set(handles.edAdjust,'string',Save.edAdjust);
    set(handles.edSumFrameStep,'string',Save.edSumFrameStep);
    set(handles.edFrameStep,'string',Save.edFrameStep);
% Loading ThetaPhiR structure
    handles.position = Save.position;
    handles.ThetaPhiR = Save.ThetaPhiR;
    handles.Bw = Save.Bw;
    Draw(hObject,handles)


% --- Executes on button press in pbSaveParam.
function pbSaveParam_Callback(hObject, eventdata, handles)
% hObject    handle to pbSaveParam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Save.S = handles.S; % save s structure;
% Saving checkbox structure
        Save.rbR_value = get(handles.rbR,'value');
        Save.rbG_value = get(handles.rbG,'value');
        Save.rbB_value = get(handles.rbB,'value');
    %---
        Save.chR_value = get(handles.chR,'value');
        Save.chG_value = get(handles.chG,'value');
        Save.chB_value = get(handles.chB,'value');
    %---
        Save.chAdjust  = get(handles.chAdjust,'value');
        Save.chSight   = get(handles.chSight,'value');
%         Save.chSumFrames = get(handles.chSumFrames,'value');
% Saving lambda edit boxes
    Save.edR = get(handles.edR,'string');
    Save.edG = get(handles.edG,'string');
    Save.edB = get(handles.edB,'string');
% Saving position paramiters
    Save.Pk    = get(handles.edPdrop,'string');
    Save.edCCD = get(handles.edCCD,'string');
    Save.edAperture = get(handles.edAperture,'string');
    Save.shLW     = get(handles.edLineSh,'string');
    Save.ed_Sh_l1 = get(handles.ed_Sh_l1,'string'); % distance between lenses
% Saving Frame step and Adjust box
    Save.edAdjust = get(handles.edAdjust,'string');
    Save.edSumFrameStep = get(handles.edSumFrameStep,'string');
    Save.edFrameStep = get(handles.edFrameStep,'string');
% Saving ThetaPhiR and mask
    Save.Bw = handles.Bw;
    Save.position = handles.position;
    Save.ThetaPhiR = handles.ThetaPhiR;
    assignin('base','Save',Save);

function ed_Sh_l1_Callback(hObject, eventdata, handles)
% hObject    handle to ed_Sh_l1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Draw(hObject,handles);
% Hints: get(hObject,'String') returns contents of ed_Sh_l1 as text
%        str2double(get(hObject,'String')) returns contents of ed_Sh_l1 as a double


% --- Executes during object creation, after setting all properties.
function ed_Sh_l1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_Sh_l1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbNA_Aprox.
function pbNA_Aprox_Callback(hObject, eventdata, handles)
% hObject    handle to pbNA_Aprox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Bw2 = roipoly(handles.cF(:,:,1),handles.R1(1,:)+ handles.TshX + handles.shX,handles.R1(2,:)+ handles.TshY + handles.shY);
Ap = sum(sum(Bw2,2)~=0);  % The aperture (diameter) size in [pix]
Apm = Ap*handles.S.PixSize; % The aperture (diameter) size in [mm]
Phi = atan(handles.S.dH/2/handles.S.R_dis_Ring);
LCCD = Apm/tan(Phi)/2; % The distance from lens to CCD element
% compound of masks   
id = find(sum(handles.Bw,1)); % id - non-zero pixels
NA_Theta = atan((id-320+ handles.TshX + handles.shX)*handles.S.PixSize/LCCD);
I = zeros( handles.N_frames, length(NA_Theta) );
nom = 1;
 wb = waitbar(0,'Angle Calculation');
 t=0;
for i = 1 : length( handles.f ) % cycle by pieces of movie
    handles.fn = [handles.dir handles.f{i}];
    inf = aviinfo(handles.fn);
    for j = 1 : inf.NumFrames
        s = sprintf('Processing %d-th frame. \n Estimated time -> %f[min]; time left %f',...
                     j,handles.N_frames*t/60,(handles.N_frames-nom)*t/60);
        waitbar(i/length( handles.f ),wb,s);
        %  Choosing the channel
        handles.nom = j;
        handles.cF = FrameRider(hObject,handles);
        if get(handles.rbR,'value')
            Frame = handles.cF(:,:,1);
        end
        if get(handles.rbG,'value')
            Frame = handles.cF(:,:,2);
        end
        if get(handles.rbB,'value')
            Frame = handles.cF(:,:,3);
        end
       tic; 
     Temp =  mean(Frame.*handles.Bw,1);
      I(nom,:) =Temp(id);
      t = toc;
      nom = nom + 1;
    end
end
close(wb);
assignin('base','I_NA',I);
assignin('base','Theta_NA',NA_Theta)


% --------------------------------------------------------------------
function muCI_Callback(hObject, eventdata, handles)
% hObject    handle to muCI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function muCalcIm_Callback(hObject, eventdata, handles)
% hObject    handle to muCalcIm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
% IM = evalin('base','IM_Mie');
% IC = evalin('base','IM_Cor');
% handles.NP = 700;
% Theta = linspace(min(handles.ThetaPhiR(:,1)),max(handles.ThetaPhiR(:,1)),handles.NP);
%  ip = find( handles.Bw ); % indexes of good points
%             Frame = IM./IC;
%       sf = smooth(handles.ThetaPhiR(:,1),  Frame(ip),50);
%       cf = fit(handles.ThetaPhiR(:,1), sf,'smooth');
%       I = cf(Theta);
% assignin('base','IC',I);
% assignin('base','ThetaC',Theta);
%% Creation of electrode border
handles.S.N = 2e3;
Br = single(BorderCreation(hObject,handles));
% figure;
% get only the part of border points 
Vb = (1 + handles.S.N ) : handles.S.N * 2;

%
% calculation of angles for border points
% Creation angle vector
%         Theta = zeros(1,length(Vb));
%         nx = [1, 0, 0];  % x ort vector
%         for i = 1:length(Vb)
%            Bv =  Br(Vb(i),:) - handles.S.Pk;
%            Bv(3) = 0;
%            Bv = Bv./norm(Bv);
%            Theta(i) = acos(dot(nx,Bv))*sign(Bv(2));
%         end
% % Shift theta to the right position
%         Theta = Theta+pi/2;

%
% Pattrns generation for that range of angles
%  
% Generation  of pattern:
 % droplet parameters;
%        r = 10e3;
%        m = 1.45;
%      % laser parameters  
%        waves.wavelength   = 458;
%        waves.theta        = 0; 
%        waves.polarization = 0;
% 
%         It = GeneratePattern(r, m, Theta, waves);
%         figure; plot(Theta*180/pi,It) 
% Moving across z coordinate
%         figure;
%         hold on;
%         VH = linspace(handles.S.dH/2,-handles.S.dH/2,handles.S.N);
%         for j = 1:50:handles.S.N
%             plot3( Br(Vb,1),Br(Vb,2),ones(size(Br(Vb,2)))+VH(j));
%         end
%          hold off;

IM = single(zeros(480,640));
Dw = linspace(-handles.S.dW/2,handles.S.dW/2,handles.S.N);
Dh = linspace(-handles.S.dH/2,handles.S.dH/2,handles.S.N);
VH=single(zeros(handles.S.N));
VH = linspace(handles.S.dH/2,-handles.S.dH/2,handles.S.N);
% 
% I0 = sind(linspace(0,4*360,5500)).^2;
hwb = waitbar(0,'Computation of intensity matrix ...');
disp(handles.S.GPU)
if(handles.S.GPU)
%    disp(handles.S.m2)
    disp(handles.S.GPU)
    tic;
    %[P,IM]=RayTracingCUDA(Br,Vb,VH,handles);
    [P,IM]=RayTracingCUDA(Br(Vb(:),1),Br(Vb(:),2),VH,handles);
    disp(toc)
    waitbar(1,hwb)
    %disp(P(:,:,1000:1010,1000))
    %disp(P(:,:,1:10,1:10))
else
for i = 1 : length(VH)    % The movement across Z axis
    waitbar(i/length(Dw),hwb)
    for j = 1 : length(Vb) % The movement across curved border
        Pd = [ Br(Vb(j),1), Br(Vb(j),2), VH(i) ]; % toczka na kraju diafragmy
        P = RayTracing(Pd,handles.S);
         if size(P,1) == 7
             dist = norm(P(1,:)-P(2,:))+norm(P(2,:)-P(3,:))+...
                    norm(P(3,:)-P(4,:))+norm(P(4,:)-P(5,:))+...
                    norm(P(5,:)-P(6,:))+norm(P(6,:)-P(7,:));
             vR = P(7,:)-P(6,:);
             vR = vR./norm(vR);
             alp = acos(dot([1,0,0],vR));
                 W =  handles.shX + ( handles.S.CCDW/2 +P(end,2))/handles.S.PixSize;
                 Hi =  handles.shY + (handles.S.CCDH/2 +P(end,3))/handles.S.PixSize;
                 IM(round(Hi),round(W)) = IM(round(Hi),round(W)) + 1*cos(alp)./(dist^2); % It(j)*cos(alp)./(dist^2); % +I0(i); %
         end
    end
end
end
assignin('base','IM_Cor',IM);
imtool( IM )
close(hwb)
% save('RayTrasing_I_correction.mat','IM');
 



function edFrStep_Callback(hObject, eventdata, handles)
% hObject    handle to edFrStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edFrStep as text
%        str2double(get(hObject,'String')) returns contents of edFrStep as a double


% --- Executes during object creation, after setting all properties.
function edFrStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edFrStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edFrameStep_Callback(hObject, eventdata, handles)
% hObject    handle to edFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edFrameStep as text
%        str2double(get(hObject,'String')) returns contents of edFrameStep as a double


% --- Executes during object creation, after setting all properties.
function edFrameStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edFrameStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbEditMask.
function pbEditMask_Callback(hObject, eventdata, handles)
% hObject    handle to pbEditMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.chR,'value') % mask for red channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,1) );
    set(hf,'name','Set Mask for Red channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(1),'xdata'),...
        get(handles.hl(1),'ydata'),'r');
    % Start to drawing matrix
    h = impoly(ha,handles.R_position); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWR = Bw1.*Bw2;
    handles.R_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end

if get(handles.chG,'value') % mask for green channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,2) );
    set(hf,'name','Set Mask for Green channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(4),'xdata'),...
        get(handles.hl(4),'ydata'),'g');
    % Start to drawing matrix
    h = impoly(ha,handles.G_position); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWG = Bw1.*Bw2;
    handles.G_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end

if get(handles.chB,'value') % mask for blue channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,3) );
    set(hf,'name','Set Mask for Blue channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(7),'xdata'),...
        get(handles.hl(7),'ydata'),'b');
    % Start to drawing matrix
    h = impoly(ha,handles.B_position); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWB = Bw1.*Bw2;
    handles.B_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end

% --- Executes on button press in pbSetMask.

 function pbSetMask_Callback(hObject, eventdata, handles)
% hObject    handle to pbSetMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Setting the mask 
if get(handles.chR,'value') % mask for red channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,1) );
    set(hf,'name','Set Mask for Red channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(1),'xdata'),...
        get(handles.hl(1),'ydata'),'r');
    % Start to drawing matrix
    h = impoly(ha); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWR = Bw1.*Bw2;
    handles.R_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end
   
if get(handles.chG,'value') % mask for green channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,2) );
    set(hf,'name','Set Mask for Green channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(4),'xdata'),...
        get(handles.hl(4),'ydata'),'g');
    % Start to drawing matrix
    h = impoly(ha); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWG = Bw1.*Bw2;
    handles.G_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end

if get(handles.chB,'value') % mask for blue channel
    % Prepearing image
    hf = imtool( handles.cF(:,:,3) );
    set(hf,'name','Set Mask for Blue channel!')
    ha = get(hf,'CurrentAxes');
    hold(ha,'on');
    plot(ha,get(handles.hl(7),'xdata'),...
        get(handles.hl(7),'ydata'),'b');
    % Start to drawing matrix
    h = impoly(ha); % Create draggable, resizable polygon
    position = wait(h);
    delete(hf);
    Bw1 = roipoly(handles.cF(:,:,1),[position(:,1).' position(1,1)],[position(:,2).' position(1,2)]);
    % mask from aperture
    Bw2 = roipoly(handles.cF(:,:,1),get(handles.hl(1),'xdata'),get(handles.hl(1),'ydata'));
    % compound of masks  for red channel
    handles.BWB = Bw1.*Bw2;
    handles.B_position = position; % coord handles for mascksinates of mask
    guidata(hObject,handles);
end
   
   



% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function muIntCon_Callback(hObject, eventdata, handles)
% hObject    handle to muIntCon (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    if get(handles.chR,'value')
        Frame = handles.cF(:,:,1);
        s='Red Frame';
         hf = imtool(Frame);
    title(s);
    figure;
    mesh(Frame);
    title(s);
    end
    if get(handles.chG,'value')
        Frame = handles.cF(:,:,2);
        s='Green Frame';
         hf = imtool(Frame);
    title(s);
    figure;
    mesh(Frame);
    title(s);
    end
    if get(handles.chB,'value')
        Frame = handles.cF(:,:,3);
        s='Blue Frame';
         hf = imtool(Frame);
    title(s);
    figure;
    mesh(Frame);
    title(s);
    end
    
% --------------------------------------------------------------------
function muCros_Callback(hObject, eventdata, handles)
% hObject    handle to muCros (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function muFitModel_Callback(hObject, eventdata, handles)
% hObject    handle to muFitModel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.S.N = 2e2;
Br = BorderCreation(hObject,handles);
% figure;
% get only the part of border points 
Vb = (1 + handles.S.N ) : handles.S.N * 2;
IM = zeros(480,640);
Dw = linspace(-handles.S.dW/2,handles.S.dW/2,handles.S.N);
Dh = linspace(-handles.S.dH/2,handles.S.dH/2,handles.S.N);
VH = linspace(handles.S.dH/2,-handles.S.dH/2,handles.S.N);
% 
% I0 = sind(linspace(0,4*360,5500)).^2;
hwb = waitbar(0,'Computation of intensity matrix ...');

InSurph = zeros(length(VH), length(Vb),3);
OutSurph = InSurph;

for i = 1 : length(VH)    % The movement across Z axis
    waitbar(i/length(Dw),hwb)
    for j = 1 : length(Vb) % The movement across curved border
        Pd = [ Br(Vb(j),1), Br(Vb(j),2), VH(i) ]; % toczka na kraju diafragmy
        P = RayTracing(Pd,handles.S);
         if size(P,1) == 7
            InSurph(i,j,:) = Pd;
            OutSurph(i,j,:) = P(7,:);
         end
    end
end

close(hwb)


% --- Executes on button press in chGPU.
function chGPU_Callback(hObject, eventdata, handles)
% hObject    handle to chGPU (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chGPU
handles.S.GPU=get(hObject,'Value');
guidata(hObject,handles);
disp(handles.S.GPU)
