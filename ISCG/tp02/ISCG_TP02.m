% Creation of the animation the Head says Yes 
%
% Author F. Marin 
% date 30/09/2019 
% version 1.1

%% RAZ
clear
close all 
clc 

%% open an read CSV 
    
    % read stl file
    [FacesHead,VerticesHead] = stlread('Facebin.stl ');
    
    % modification des axes
     VerticesHead2=[VerticesHead(:,1),VerticesHead(:,3),VerticesHead(:,2)];
     
    % determination of number of vectices and faces  
     nVerticesHead=size(VerticesHead2,1);
     nFacesHead=size(FacesHead,1);

%% Definition of the rotation 
    
    % finition of the axis of rotation
        AxisRot=[1;0;0]; % the norm could be different to 1
        PAxisRot_lab=[0;-75;-75]; % point on the axis of rotation 
        e_lab=AxisRot/norm(AxisRot); %unit vector of axis of rotation 
    
     % defitnion of cpm of unit vector
            E=[0,-e_lab(3),e_lab(2);...
            -e_lab(3),0,-e_lab(1);...
            -e_lab(2),e_lab(1),0]; 
    
     % definition of the angle of rotation    
         
        RoMRotation=20; % range of motion of the rotation 
        pasRotation=1;  % angle step between to rotation 
   
        % creation of the vector of the angle of rotation 
        AnglesRotation=[0:pasRotation:RoMRotation,...
                        RoMRotation:-pasRotation:-RoMRotation,...
                        -RoMRotation:pasRotation:0];
    
        % calculation of the number of fram 
        nframe=size(AnglesRotation,2);

%% Create the animation 

 % Figure opening 
    fig1=figure(1); 
    
% Defition of  axes
     axis([-100,100,-200,100,-150,150]);
     view(-140,4)
            % modification of axes properties 
             ax1=gca;
             ax1.NextPlot = 'replaceChildren';
             ax1.DataAspectRatio=[1,1,1]; 
             ax1.DataAspectRatioMode='manual';

%  draw the initiale position of the  head     
   
   %defintion of the ligth 
        light('Position',[.1 .5 1],'Style','infinite')
   
   % Faces color definition 
            ColorFaces=[.6,.7,.5]; 
            
   % draw the head as a patch 
     Head3D=patch('Faces',FacesHead,'Vertices',VerticesHead2);
     % update of the patch properties 
        Head3D.FaceVertexCData=ColorFaces;
        Head3D.FaceColor='flat';
        Head3D.LineStyle='none';
        Head3D.FaceLighting='phong';

% Creation of the animation 
    %predefinition of the movie object HeadMovie
        HeadMovieFrames(nframe)=struct('cdata',[],'colormap',[]);
    
    % creation de the movie file 
        HeadMovie=VideoWriter('HeadMovie3','MPEG-4');
        HeadMovie.FrameRate=50;
        HeadMovie.Quality=100;
        open(HeadMovie);
    
     % determination of the frame    
        for i=1:nframe 
    
            %definition of rotation 
            Rot=eye(3,3)+sind(AnglesRotation(i))*E+(1-cosd(AnglesRotation(i)))*E*E;
            vT=(eye(3,3)-Rot)*PAxisRot_lab;
          
            %rotation of Vectices ( be careful row vector formalism) 
                for j=1:nVerticesHead
                    VerticesHeadRot(j,:)=(Rot*VerticesHead2(j,:)')'+((eye(3)-Rot)*PAxisRot_lab)';
                end 
  
    % update 3D object  
    Head3D.Vertices=VerticesHeadRot;
    drawnow 
    
    % Storage of the frame 
    HeadMovieFrames(i)=getframe; 
   
    % write the frame into the movie file
    writeVideo(HeadMovie,HeadMovieFrames(i));

        end 
    
   % closure of the movie frame 
   close(HeadMovie)

%% END 