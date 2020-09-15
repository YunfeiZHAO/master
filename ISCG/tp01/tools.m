%% definition of useful functions

classdef tools
    methods(Static)
    function p = plot_point(p, marker, color, size)
        p = plot3(p(1), p(2), p(3));
        p.Marker=marker;
        p.MarkerFaceColor=color;
        p.MarkerSize=size;
    end
    
    % plot vector p1p2
    function plot_vector(p1, p2, color)
        quiver3(p1(1), p1(2), p1(3), p2(1), p2(2), p2(3), 'color', color);
    end
    
    % add text on a point
    function point_add_text(p, offset, content)
        text(p(1) + offset, p(2) + offset, p(3) + offset, content);
    end
    
    function l = plot_line(p1, p2, color, width)
        l = line([p1(1), p2(1)],...
                 [p1(2), p2(2)],...
                 [p1(3), p2(3)]);
        l.Color = color;
        l.LineWidth = width;
    end

    end 
end



