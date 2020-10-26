show_trajectory([r1,r2,r3,r4,r5]);

function tj = show_trajectory(rs)
    % create figure
    c = ['k', 'b', 'g', 'r', 'm']
    w = [1,1,1,1,1]
    for i = 1:5
        r = rs(i);
        [x, y] = r.signals.values;
        plot(x,y,'linewidth',w(i),'color',c(i));
        hold on;
    end
    legend('robot1','robot2','robot3','robot4','robot5')
    title('Trajectory of robots of 5 connections for the second destination')
    xlabel('abscissa X') 
    ylabel('ordinate Y') 
end

