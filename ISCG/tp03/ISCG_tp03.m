% Creation of the animation the Head says Yes 
%
% Author Yunfei ZHAO
% date 20/10/2020
% version 1.1

%% RAZ
clear
close all 
clc

%% open an read CSV 
    
    % read stl file
    [FacesMan,VerticesMan] = stlread('FlyingMan.stl');
     
    % determination of number of vectices and faces  
     nVerticesMan=size(VerticesMan,1);
     nFacesMan=size(FacesMan,1);

     
%% Draw FlyingMan 

 % Figure opening 
    fig1=figure(1); 
            
   % draw the head as a patch 
     FM3D=patch('Faces',FacesMan,'Vertices',VerticesMan,'Facelighting','gouraud');
     % update of the patch properties 
        FM3D.FaceColor=[1 0 0];
        FM3D.EdgeColor='none';
        FM3D.AmbientStrength=0.15;
        
    
%% Scrow motion calculation
    % unit vector of the screw
    u = [1;2;3];
    e = u/norm(u);
    % location of the screw
    A = [1,2,3];
    % screw angle
    theta = 45;
    % screw translation
    d = 15;
    % step
    nstep = 10;
    %screw angles steps
    theta_step = ([0: nstep]/nstep)*theta;
    % screw translation steps
    d_step=([0:nstep]/nstep)*d
    
% E matrix calculation
    E = [0, -e(3), e(2);...
         e(3), 0, -e(1);...
         -e(2), e(1), 0];


% Do the loop for animation
for i=1:nstep+1
   % rotation matrix
   R = eye(3) + sind(theta_step(i))*E + (1-cosd(theta_step(i)))*E*E;
   %translation
   D = ((eye(3) - R)*A')' + d_step(i)*e';
   %screw motion og the points ion
   for j=1:nVerticesMan
       VerticesMan_i(j, :) = (R*VerticesMan(j,:)')' + D;
   end
      % draw the head as a patch 
   FM3D=patch('Faces',FacesMan,'Vertices',VerticesMan_i,'Facelighting','gouraud');
     % update of the patch properties 
        FM3D.FaceColor=[1 0 0];
        FM3D.EdgeColor='none';
        FM3D.AmbientStrength=0.15;
end   
   
axis image





