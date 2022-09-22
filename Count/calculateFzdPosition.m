function [ flag ] = calculateFzdPosition( x1,y1,x2,y2,x3,y3 )


S = (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3);
if S>0
    flag = 1;
elseif S<0
    flag = -1;
else
    flag = 0;
end


end

