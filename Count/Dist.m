function [ d ] = Dist( x0,y0,x1,y1,x2,y2 )

    d=abs((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0))/sqrt((x1-x2)^2+(y1-y2)^2);

end

