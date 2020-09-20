%% ISCG_practice.m
%
% application of matlab for the first homework
% author : Yunfei ZHAO
% date : 20/9/2020
% version 1
%% RAZ

close all
clear all
clc

addpath('/Users/yunfei/Desktop/master/ISCG');

%% add reference of the camera
% origin
O_R0 = [0;0;0]; % it is a colonne vectors 3 * 1 matrix; [x,y,z]: line vector 1 * 3

% Axis of Ro
x0_R0 = [1;0;0];
y0_R0 = [0;1;0];
z0_R0 = [0;0;1];

%% Read CSV file
drone_data = csvread('DataMarkersDrone.csv', 2);

% load X, Y, Z in CM(camera) reference
[n, c] = size(drone_data);
t = drone_data(:, 1);
X_C_CM = drone_data(:, 2);
Y_C_CM = drone_data(:, 3);
Z_C_CM = drone_data(:, 4);
X_R1_CM = drone_data(:, 5);
Y_R1_CM = drone_data(:, 6);
Z_R1_CM = drone_data(:, 7);
X_R2_CM = drone_data(:, 8);
Y_R2_CM = drone_data(:, 9);
Z_R2_CM = drone_data(:, 10);
X_R3_CM = drone_data(:, 11);
Y_R3_CM = drone_data(:, 12);
Z_R3_CM = drone_data(:, 13);
X_R4_CM = drone_data(:, 14);
Y_R4_CM = drone_data(:, 15);
Z_R4_CM = drone_data(:, 16);

% Open figure
figure(1);
% add object
hold on;
% add grid
grid on;
% square grid(make image adapte to data)
axis image;

% plot origin
tools.plot_point(O_R0, 'o', 'k', 10);
% plot Axis
tools.plot_vector(O_R0, x0_R0, 'b');
tools.plot_vector(O_R0, y0_R0, 'r');
tools.plot_vector(O_R0, z0_R0, 'g');

%% plot trajectory points in CM reference
p = scatter3(X_C_CM, Y_C_CM, Z_C_CM, 'o', 'r');
pR1 = scatter3(X_R1_CM, Y_R1_CM, Z_R1_CM, 'o', 'b');
pR2 = scatter3(X_R2_CM, Y_R2_CM, Z_R2_CM, 'o', 'b');
pR3 = scatter3(X_R3_CM, Y_R3_CM, Z_R3_CM, 'o', 'b');
pR4 = scatter3(X_R4_CM, Y_R4_CM, Z_R4_CM, 'o', 'b');
quiver3(X_C_CM(1:n-1), Y_C_CM(1:n-1), Z_C_CM(1:n-1), X_C_CM(2:n) - X_C_CM(1:n-1), Y_C_CM(2:n) - Y_C_CM(1:n-1), Z_C_CM(2:n) - Z_C_CM(1:n-1), 'color', 'b');


%% plot the DR (drone deplacement) in the camera reference
% Open figure
figure(2);
% add object
hold on;
% add grid
grid on;
% square grid(make image adapte to data)
axis image;

% plot origin
tools.plot_point(O_R0, 'o', 'k', 10);
% plot Axis
tools.plot_vector(O_R0, x0_R0, 'b');
tools.plot_vector(O_R0, y0_R0, 'r');
tools.plot_vector(O_R0, z0_R0, 'g');

% Define reference frame RD (reference of drone)
% we build a reference from R3

% definitio of intermediare vectors
R3R1_CM = [X_R1_CM - X_R3_CM, Y_R1_CM - Y_R3_CM, Y_R1_CM - Y_R3_CM];
R3R4_CM = [X_R1_CM - X_R4_CM, Y_R1_CM - Y_R4_CM, Y_R1_CM - Y_R4_CM];

% calculation of the axis of RD
x1_RD = R3R1_CM ./ vecnorm(R3R1_CM, 2, 2);
z1_RD = cross(R3R1_CM, R3R4_CM) ./ vecnorm(cross(R3R1_CM, R3R4_CM), 2, 2);
y1_RD = cross(z1_RD, x1_RD);
quiver3(X_R3_CM, Y_R3_CM, Z_R3_CM, x1_RD(:, 1), x1_RD(:, 2), x1_RD(:, 3), 'color', 'b');
quiver3(X_R3_CM, Y_R3_CM, Z_R3_CM, y1_RD(:, 1), y1_RD(:, 2), y1_RD(:, 3), 'color', 'r');
quiver3(X_R3_CM, Y_R3_CM, Z_R3_CM, z1_RD(:, 1), z1_RD(:, 2), z1_RD(:, 3), 'color', 'g');


%% plot displacement of the point C in drone's coodinate system
% Open figure
figure(3);
% add object
hold on;
% add grid
grid on;
% square grid(make image adapte to data)
axis image;

% plot origin
tools.plot_point(O_R0, 'o', 'k', 10);
% plot Axis
tools.plot_vector(O_R0, x0_R0, 'b');
tools.plot_vector(O_R0, y0_R0, 'r');
tools.plot_vector(O_R0, z0_R0, 'g');

C_RD = zeros(n, 3);
for i = 1:n
    % definition of transfer matrix from RD to CM
    T_CM_RD = [x1_RD(i, :).', y1_RD(i, :).', z1_RD(i, :).'];
    % definition of transfer matrix from CM to CM
    T_RD_CM = transpose(T_CM_RD);
    % calculation of point C in RD (Drone reference)
    C_CM = [X_C_CM(i); Y_C_CM(i); Z_C_CM(i)];
    R3_CM = [X_R3_CM(i); Y_R3_CM(i); Z_R3_CM(i)];
    C_RD(i,:) = T_RD_CM * (C_CM - R3_CM);
end

scatter3(C_RD(:, 1), C_RD(:, 2), C_RD(:, 3), 'o', 'g');
var = max(vecnorm(C_RD, 2, 2)) - min(vecnorm(C_RD, 2, 2));








