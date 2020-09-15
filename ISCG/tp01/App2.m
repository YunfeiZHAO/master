%% Define reference frame R1

close all
clear all
clc


P1_R0 = [5;6;10];
P2_R0 = [9;8;3];
P3_R0 = [2;2;7];

% define vectors P1P2 P1P3
v_R0 = P2_R0 - P1_R0;
u_R0 = P3_R0 - P1_R0;

% calculate if the axes of R1
    x1_R0 = v_R0/norm(v_R0);
    z1_R0 = cross(v_R0, u_R0)/norm(cross(v_R0, u_R0));
    y1_R0 = cross(z1_R0, X1_R0);
% draw R1 change
    quiver3(O_R0  (1), O_R0(2), O_R0(3), x0_R0(1), x0_R0(2), x0_R0(3));
    quiver3(O_R0(1), O_R0(2), O_R0(3), y0_R0(1), y0_R0(2), y0_R0(3));
    quiver3(O_R0(1), O_R0(2), O_R0(3), z0_R0(1), z0_R0(2), z0_R0(3));

%% calculation of P2 in R1
    % define transform matrix from R1 to R0
    T_R1_R0 = [x1_R0, y1_R0, R0];
    
    % calculation of the transfer matrix from R0 to R1
    T_R0_R1 = transpose(T_R1_R0)
    
    % calculation of P2 in R1
    P2_R1 = T_R1_R0 * (P2_R0 - P1_R0)
    
    % calculation off P3 in R1
    P3_R1 = T_R1_R0 * (P3_R0 - P1_R0)


