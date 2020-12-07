%toolkit for static robot localization computation
clc; clear; close

%beacons
xb=[-5;-5; 5;5];
yb=[-5; 5;-5;5];
%real position (x,y) of the robot
xr=[0;0];

figure(1);hold on
plot(xb,yb,'*');
axis([-6 6 -6 6]);
plot(xr(1),xr(2),'or')
title('Real position of the robot');

%measurements
% distance mesured by telemeter with noise
randn('seed',100)
r = sqrt((yb-xr(2)).^2+(xb-xr(1)).^2) + randn(length(xb), 1);
% Initial guess
x = [6, 6];
% newton iteration
for i = 1:10
    H = jacob(x, xb, yb);
    xnew = x + H\dy(x, r, xb, yb);
    if(norm(x - xnew) < 1e-5) 
        break;
    end
    x = xnew;
    plot(x(1),x(2),'og')
end

plot(x(1),x(2),'ob')


% residual computation
function out = dy(x, r, xb, yb)
    out = r.^2 - ((yb-x(2)).^2+(xb-x(1)).^2);
end
% Jacobian computation
function H = jacob(x, xb, yb)
    H = [2*(x(1)-xb), 2*(x(2)-yb)];
end



