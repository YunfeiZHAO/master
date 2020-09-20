%% ISCG_practice.m
%
% application of matlab during ISCG lectures
% author : Yunfei ZHAO
% date : 15/9/2020
% version 1
%% RAZ

close all
clear all
clc

%% creation of R0 (lab reference frame)
    % definition of the reference frame R0
    % origin
    O_R0 = [0;0;0]; % it is a colonne vectors 3 * 1 matrix; [x,y,z]: line vector 1 * 3
    
    % Axis of Ro
    x0_R0 = [1;0;0];
    y0_R0 = [0;1;0];
    z0_R0 = [0;0;1];
    
    % Points
    P1_R0 = [5;6;10];
    P2_R0 = [9;8;3];
    P3_R0 = [2;2;7];
    % Open figure
    figure(1);

    % add object
    hold on;

    % add grid
    grid on;

    % plot origin
    tools.plot_point(O_R0, 'o', 'k', 10);

    % plot Axis
    tools.plot_vector(O_R0, x0_R0, 'b');
    tools.plot_vector(O_R0, y0_R0, 'r');
    tools.plot_vector(O_R0, z0_R0, 'g');

    % square grid(make image adapte to data)
    axis image;
    % name of grid (add label to grid)
    xlabel(['Xaxe']);
    ylabel(['Yaxe']);
    zlabel(['Zaxe']);

    % plot P1 P2
    tools.plot_point(P1_R0, 'o', 'y', 5);
    tools.plot_point(P2_R0, 'o', 'r', 5);
    % legend   
    offsettext = 0.2;
    tools.point_add_text(P1_R0, offsettext, ['P1']);
    tools.point_add_text(P2_R0, offsettext, ['P2']);

    % plot line P1 P2
    tools.plot_line(P1_R0, P2_R0, 'r', 5)

    % compute the distance
    v_R0 = P2_R0 - P1_R0;
    d1 = norm(v_R0);
    m = (P1_R0 + P2_R0)/2;
    tools.point_add_text(m, offsettext, ['DistanceP_1P_2=', num2str(d1)]);

    % compute the unit vector of v_R0
    vunit_R0 = v_R0/norm(v_R0);

%% Define referece frame R1
    % definition of intermediare vectors
    v_R0 = P2_R0 - P1_R0;
    u_R0 = P3_R0 - P1_R0;
    
    % calculation of the axis of R1
    x1_R0 = v_R0/norm(v_R0);
    z1_R0 = cross(u_R0, v_R0)/norm(cross(u_R0, v_R0));
    y1_R0 = cross(z1_R0, x1_R0); % because x1_R0 and y1_R0 is unit vector and othogonal so don't need to divide the norm
    
    % draw R1
    tools.plot_vector(P1_R0, x1_R0, 'b');
    tools.plot_vector(P1_R0, y1_R0, 'r');
    tools.plot_vector(P1_R0, z1_R0, 'g');
    
%% calculation of P2, P3 in R1
    % definition of transfer matrix from R1 to R0
    T_R0_R1 = [x1_R0, y1_R0, z1_R0];
    % definition of transfer matrix from R0 to R1
    T_R1_R0 = transpose(T_R0_R1); % because this matrix is orthogonal, m.t = x.inverse
    % calculation of P2 in R1
    P2_R1 = T_R1_R0 * (P2_R0 - P1_R0);
    % calculation of P3 in R1
    P3_R1 = T_R1_R0 * (P3_R0 - P1_R0);
    disp(P2_R1);
    disp(P3_R1);
    
        
        
        
        

