clc
close all
clear all

trainPath='result_images\';    
theFiles  = dir([trainPath '*.png']);
disp(length(theFiles));
count = 1;
train_num = length(theFiles);
sort_nat_name=sort_nat({theFiles.name}); 

%read data
HT = xlsread('HeadTail.csv');
Ph = xlsread('Pharynx.csv');
PP = xlsread('PeakPoints.csv');
IP = xlsread('InflectionPoints.csv');

head = HT(:,1:2);
tail = HT(:,3:4);
pharynx = Ph(:,1:2);

[m,n] = size(PP);
[p,q] = size(IP);

Data = zeros(100,5);

anglechange=[];
maxDist=[];

for k = 1:train_num
    fullFileName = sort_nat_name{k};
    I = imread([trainPath fullFileName]);
    count = count +1;

    set(0,'DefaultFigureVisible', 'off')
    
    PPnum = 0;
    IPnum = 0;
    
    for n1=1:n
        if isnan( PP(k,n1) )
            PPnum = PPnum+1;
        end
    end
    for n2=1:q
        if isnan( IP(k,n2) )
            IPnum = IPnum+1;
        end
    end
    
    PPnum1 = n - PPnum;
    IPnum1 = q - IPnum;
 
    %%Error checking mechanism
    if ( PPnum1 +2 ) ~= IPnum1
        fprintf('not consist %d\n', k);
    end
    
    angles = [];
    dists = [];
 
    for i = 1:2:n-1
        if isnan( PP(k,i) )
            break;
        end
        
        flag = calculateFzdPosition(pharynx(k,1),pharynx(k,2),tail(k,1),tail(k,2),PP(k,i),PP(k,i+1));
        angle = bendAngle( IP(k,i),IP(k,i+1),PP(k,i),PP(k,i+1),IP(k,i+2),IP(k,i+3) );
        angle = angle * flag;
        angle=roundn(angle,-2);
          
        dist = Dist( PP(k,i),PP(k,i+1),pharynx(k,1),pharynx(k,2),tail(k,1),tail(k,2) ); 
        dist = dist * flag;
        angles = [angles angle];
        dists = [dists dist];       
        
    end
    Data(k,1:length(angles)) = angles;
    Datadist(k,1:length(dists)) = dists;
    
    a=length(angles);
    min = abs( angles(1) );
    index = 1;
   
    for t=1:a
        if min > abs( angles(t) )
            min = abs( angles(t) );
            index = t;
        end
    end

    anglechange = [anglechange angles(index)];

    b=length(dists);
    max = abs( dists(1) );
    index1 = 1;
    
    for t=1:b
        if max < abs( dists(t) )
            max = abs( dists(t) );
            index1 = t;
        end
    end

    maxDist = [maxDist dists(index)];

     for i = 1:2:q-1
        if isnan( IP(k,i) )
            break;
        end
     end   
  
end

bodyBendnum = 0;
for i = 1:length(anglechange)-1
    if sign( anglechange(i) ) *sign(  anglechange(i+1) )  == -1
        if i-3>=1 && i+4<=length(maxDist)
            s1 = sign( anglechange(i) ) *sign(  anglechange(i-1) );
            
            s4 = sign( anglechange(i+1) ) *sign(  anglechange(i+2) );
           
            if s1 == 1  && s4 == 1
                bodyBendnum = bodyBendnum + 1;
            end
            
        end
    end
end
fprintf('Angle: the number of body bends are %d\n', bodyBendnum) 


bodyBendnum1 = 0;

for i = 1:length(maxDist)-1
    if sign( maxDist(i) ) *sign(  maxDist(i+1) )  == -1
        if i-3>=1 && i+4<=length(maxDist)
            s1 = sign( maxDist(i) ) *sign(  maxDist(i-1) );
           
            s4 = sign( maxDist(i+1) ) *sign(  maxDist(i+2) );
          
            if s1 == 1  && s4 == 1 
                bodyBendnum1 = bodyBendnum1 + 1;                 
            end
                 
        end
        
    end
end
fprintf('Dist: the number of body bends are %d\n', bodyBendnum1) 


