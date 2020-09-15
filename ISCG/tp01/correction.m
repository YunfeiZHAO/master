%% ISCG_MatlabAppication1.m 
%
%Matlab script of the Matlab Application of  ISCG lecture 
%
%Author : F.Marin 
%Date 15/09/2020
%version 1

%% RAZ

close all
clear all 
clc 

%% Creation of R0 ( lab reference frame) 

    % definition of the reference frame R0
        % origin
            O_R0=[0;0;0]; 
        % Axis of R0
            x0_R0=[1;0;0];
            y0_R0=[0;1;0];
            z0_R0=[0;0;1]; 
        
         %Open figure 
         figure(1)
         
         %add object 
         hold on 
         
         %add grid 
         grid on 
         
         %plot origin
         plotOrigin=plot3(O_R0(1),O_R0(2),O_R0(3)); 
            plotOrigin.Marker='o'; 
            plotOrigin.MarkerFaceColor='k'; 
            plotOrigin.MarkerSize=10; 
            
         %plot Axis 
          quiver3(O_R0(1),O_R0(2),O_R0(3),x0_R0(1),x0_R0(2),x0_R0(3));  
          quiver3(O_R0(1),O_R0(2),O_R0(3),y0_R0(1),y0_R0(2),y0_R0(3)); 
          quiver3(O_R0(1),O_R0(2),O_R0(3),z0_R0(1),z0_R0(2),z0_R0(3)); 
          
          %square grid 
          axis image 
          
          %name of axis 
          xlabel(['Xaxis'])
          ylabel(['Yaxis'])
          zlabel(['Zaxis'])
         
 %% plot points 
 
        %define Points
        P1_R0=[5;6;10]; 
        P2_R0=[9;8;3]; 
        
        % plot points 
        plotP1=plot3(P1_R0(1),P1_R0(2),P1_R0(3)); 
            plotP1.Marker='o'; 
            plotP1.MarkerFaceColor='b'; 
            plotP1.MarkerSize=10; 
            
        plotP2=plot3(P2_R0(1),P2_R0(2),P2_R0(3)); 
            plotP2.Marker='o'; 
            plotP2.MarkerFaceColor='r'; 
            plotP2.MarkerSize=10;  
            
         %legend of P1 and P2
         offsettext=0.2;
         text(P1_R0(1)+offsettext,...
             P1_R0(2)+offsettext,...
             P1_R0(3)+offsettext,...
             ['P_1']);
            
          text(P2_R0(1)+offsettext,...
             P2_R0(2)+offsettext,...
             P2_R0(3)+offsettext,...
             ['P_2']);  
         
 %% plot line P1 P2 
 
    %plot line 
    plotLineP1P2=line([P1_R0(1),P2_R0(1)],...
                      [P1_R0(2),P2_R0(2)],... 
                      [P1_R0(3),P2_R0(3)]); 
          plotLineP1P2.Color='k'
          plotLineP1P2.LineWidth=1.5;
          
    %compute distance P1P2
        v_R0=P2_R0-P1_R0
        d1=norm(v_R0); 
        
   % add distance on the figure 
        text((P1_R0(1)+P2_R0(1))/2+offsettext,...
             (P1_R0(2)+P2_R0(2))/2+offsettext,...
             (P1_R0(3)+P2_R0(3))/2+offsettext,...
             ['DistanceP_1P_2=',num2str(d1)]); 
         
 %% Define unit vector of v_R0 
        %v_R0=P2_R0-P1_R0
        %d1=norm(v_R0); 
        
        vunit_R0=v_R0/norm(v_R0); 
 
 %% Define reference frame R1        
                  
           % points
           P1_R0=[5;6;10]; 
           P2_R0=[9;8;3]; 
           P3_R0=[2;2;7]; 
           
           %definition of intermediate vectors
             v_R0=P2_R0-P1_R0;
             u_R0=P3_R0-P1_R0;
             
           %calculation of the axis of R1
                 x1_R0=v_R0/norm(v_R0); 
                 z1_R0=cross(v_R0,u_R0)/norm(cross(v_R0,u_R0)); 
                 y1_R0=cross(z1_R0,x1_R0); 
            
            %draw R1
            quiver3(P1_R0(1),P1_R0(2),P1_R0(3),x1_R0(1),x1_R0(2),x1_R0(3));
            quiver3(P1_R0(1),P1_R0(2),P1_R0(3),y1_R0(1),y1_R0(2),y1_R0(3));
            quiver3(P1_R0(1),P1_R0(2),P1_R0(3),z1_R0(1),z1_R0(2),z1_R0(3));
                 
  %% calculation of P2 P3 in R1 
  
    % definition transfer matrix from R1 to R0 
        T_R0_R1=[x1_R0,y1_R0,z1_R0]; 
        
    % calculation of the transfer matric frome R0 to R1    
        T_R1_R0=transpose(T_R0_R1); 
        
    % calculation of P2 in R1
        P2_R1=T_R1_R0*(P2_R0-P1_R0)
        
    % calculation of P3 in R1 
        P3_R1=T_R1_R0*(P3_R0-P1_R0)
        
        
           