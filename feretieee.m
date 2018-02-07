%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Name: Shilpashree Rao
%Email id: shilpasr@usc.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% =====================PLEASE SPECIFY THE PATH OF THE ORL DATABASE HERE======%
 %========================================================================%
path_name = 'D:.............custom feret\s';
 %========================================================================%
 %========================================================================%

global classmeanarray totalmeanarray GlobalBestP;
trainratio=8;
testratio=12;
subs=35;
trainimgs=280;
testimgs=420;
total=20;
countsum=0;
percentsum=0;
f = fspecial('gaussian',[3 3],0.5);
b = zeros(subs,trainratio);
f1 = fspecial('gaussian',[3 3],0.2);     % laplacian blur
h = fspecial('laplacian',0.5);
f2= fspecial('gaussian',[3 3],0.5);
f3= fspecial('gaussian',[3 3],0.5);
f4= fspecial('gaussian',[3 3],0.5);
zx=1;                                 % feature enhancement factor
zy=2.8;
for i= 1:trainimgs
storeface{i} = zeros(1,384);
end
%%Train and Test Iterations%
no_iter = input ('Please enter the number of iterations to be done: ');
for x = 1:no_iter           %25 Iterations
     disp(strcat('Iteration number:',num2str(x)));
    xlswrite('RR.xls', x,'Target', sprintf('A%d',x));
ttotal = zeros(1,400);
k = 1;
tic;
for j = 1:subs
%%Acquire test images Randomly 
    b(j,:) = randperm(total,trainratio);
    tsum = zeros(1,400);
        for i = 1:trainratio                      %Four images per subject    
   [face, map]=imread(strcat(path_name,...
   num2str(j),'\',num2str(b(j,i)),'.ppm'));

%%Preprocessing steps 
  face=rgb2gray(face);
  face=imresize(face,0.25);
  facetemp=im2double(face);
  face=im2double(face);
  face = imfilter(face,f,'replicate'); %Gaussian Blurr
  face = imfilter(face,h,'replicate'); %Laplacian Blurr
  face=face.*zx;                
  face=facetemp+face;                  %Enhancing the features
  face = imadjust(face,[],[],(1/zy));  %Gamma Intensity Correction
  face = imfilter(face,f1,'replicate'); %Gaussian Blurr     
  I1=imnoise(face,'salt & pepper',0.025);
  I1=medfilt2(I1);
  K1=wiener2(I1);
  face=K1;
  face = imfilter(face,f2,'replicate'); 
  face = imfilter(face,f3,'replicate'); 
  face = imfilter(face,f4,'replicate');
  
%Feature extraction%    

 [ca ch cv cd]=dwt2(face,'haar');
 [ca ch cv cd]=dwt2(ca,'haar');
%dct
 dctface=dct2(face); 
   % dctface=ca;  
    %u=dctface(1:24,1:16);                %Extract 50x50 DCT-coefficients
%Triangular-shaped feature extraction
I=dctface;
u=zeros(1,400);
radius=12;
y=zeros(radius*10);
i11=1;
i12=1;
while(i11<=radius)
    y(i11)=radius-i11+1;
    i11=i11+1;
end
length1=0;
j1=1;
for i11=1:radius
    q1=y(i11);
    u(1,j1:q1+j1-1)=I(i11,1:q1);
    length1=length1+q1;
    j1=q1+j1;
end
u=u(1,1:length1);
storeface{k} = reshape(u,1,length1);    %Store in 1x2500 vector
k = k+1;
if(i==1)
tsum = tsum(1,1:length1);
end
tsum = double(tsum)+double(u);        
end
if(j==1)
ttotal = ttotal(1,1:length1);
end
ttotal = double(tsum)+double(ttotal);    
avg = (tsum./trainratio);
classmeanarray{j} = avg;                 %Mean of DCT coefficients for 
                                         %each Class(Subject) 
end
time1=toc;
xlswrite('RR.xls', time1,'Target', sprintf('D%d',x));

avgall = ttotal/trainimgs;                        
totalmeanarray = avgall;                 %Mean of DCT coefficints of all 
                                         %Classes
                                         
%%Start BPSO                                     

%Initalization of Parameters%

NPar = length1;                             %Number of Dimensional Parameters
NumofParticles = 30;                     %Number of Particles
Velocity = zeros(NumofParticles,NPar);
Position = zeros(NumofParticles,NPar);
Cost = zeros(NumofParticles,1);
LocalBestCost = zeros(NumofParticles,1);
LocalBestPosition = zeros(NumofParticles,NPar);
ff='BPSO_FITNESS_FUNCTION';                           %Fitness function
GlobalBestP = rand(1,NPar);              
GlobalBestC = 0; 

MaxIterations = 100;                      %Number of BPSO iterations
Damp=0.6;                                %Inertial Damping Factor
C1 = 0.618;                              %Cognitive Factor
C2 = 1.618;                              %Social Factor

%Initialization of Particles%

for i = 1:NumofParticles
    Velocity(i,:) = (rand(1,NPar));
    R = rand(1,NPar);
    Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
    Cost(i,:) = feval(ff,Position(i,:),subs,NPar);
    LocalBestPosition(i,:) = Position(i,:);
    LocalBestCost(i,:) = Cost(i,:);

    if Cost(i,:) > GlobalBestC
        GlobalBestP = Position(i,:);
        GlobalBestC = Cost(i,:);
    end
end
%Start BPSO iterations
for t = 1:MaxIterations
    Damp=Damp.^t;
    for i = 1:NumofParticles
        r1 = rand(1,NPar);
        r2 = rand(1,NPar);
        w = rand(1,NPar);
        Velocity(i,:) = Damp*Velocity(i,:) + ...
            r1*C1.*(LocalBestPosition(i,:) - Position(i,:)) + ...
            r2*C2.*(GlobalBestP - Position(i,:));
         
        R = rand(1,NPar);
        Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
        Cost(i,:) =feval(ff,Position(i,:),subs,NPar);
       
        
        if Cost(i,:) > LocalBestCost(i,:);
            LocalBestPosition(i,:) = Position(i,:);
            LocalBestCost(i,:) = Cost(i,:);
            if Cost(i,:) > GlobalBestC
                GlobalBestP = Position(i,:);
                GlobalBestC = Cost(i,:);
            end
        end   
    end

 end
%End BPSO

%%Results from BPSO

count = length(find(GlobalBestP));           %Number of Features selected 
disp('Number of selected features:');
disp(count);
xlswrite('RR.xls', count,'Target', sprintf('B%d',x));
temp(x)=count;

for t= 1:trainimgs
    storeface{t}= storeface{t}.*GlobalBestP; %Feature vector for 
end                                          %each Face

%%Start Testing
tic;
rec=0;                                       %Recognition Counter
tests=testimgs;                                   %Run test for left out 420
                                             %images
for n=1:tests  
    n11=n;
    c = ceil(n/testratio); 
    b2 = 1:total;  
    b1 = setdiff(b2,b(c,:));                 %Select images not used in 
                                             %testing stages
                                             
    i = mod(n,testratio)+(testratio * (mod(n,testratio)==0));        
    
    img = imread(strcat(path_name,...
           num2str(c),'\',num2str(b1(i)),'.ppm'));
       
       img11=img;
       
%Preprocessing         
 img=rgb2gray(img);
 img=imresize(img,0.25);
 imgtemp=im2double(img);
 img=im2double(img);
 img = imfilter(img,f,'replicate');
 img = imfilter(img,h,'replicate'); %Laplacian Blurr
 img=img.*zx;
 img=imgtemp+img;
 img = imadjust(img,[],[],(1/zy));
 img = imfilter(img,f1,'replicate');
 I111=imnoise(img,'salt & pepper',0.025);
 I111=medfilt2(I111);
 K111=wiener2(I111);
 img=K111;
 img = imfilter(img,f2,'replicate');
    
%Feature Extraction
    
%    [ca ch cv cd]=dwt2(img,'haar');
%    [ca ch cv cd]=dwt2(ca,'haar');
   %[ca ch cv cd]=dwt2(ca,'haar');
   % [ca ch cv cd]=dwt2(ca ,'haar');
    pic=dct2(img);
    %pic=ca;  
    %pic_dct=reshape(pic(1:24,1:16),1,384);
%Triangular-shaped feature extraction
I=pic;
u=zeros(1,400);
radius=12;
z=zeros(radius*10);
i11=1;
while(i11<=radius)
    z(i11)=radius-i11+1;
    i11=i11+1;
end
length2=0;
j1=1;
for i11=1:radius
    q1=z(i11);
    u(1,j1:q1+j1-1)=I(i11,1:q1);
    length2=length2+q1;
    j1=j1+q1;
end
u=u(1,1:length2);    
    pic_dct=reshape(u,1,length2);
%-Feature Selection-%    
    
    opt=pic_dct.*GlobalBestP;
    
%Compute Euclidean Distance with each test vector

    d=zeros(1,trainimgs);
 
             for p=1:trainimgs 
                 r = storeface{p};
     d(p) = sqrt(sum((r-opt).^2));    
             end 
             
     [val,index1]=min(d);                   %Minimum of Euclidean Distances
                                           %gives the Matched Vector
     
     if((ceil(index1/trainratio))==c)                %Increment Recognition
     rec=rec+1;                            %Counter if successful  
     
     else
         
    n=n11;                  
    c = ceil(n/testratio); 
    img=img11;
    
%Preprocessing    
    
img=rgb2gray(img);
 img=imresize(img,0.25);
 imgtemp=im2double(img);
 img=im2double(img);
 img = imfilter(img,f,'replicate');
 img = imfilter(img,h,'replicate'); %Laplacian Blurr
 img=img.*zx;
 img=imgtemp+img;
 img = imadjust(img,[],[],(1/zy));
 img = imfilter(img,f1,'replicate');
 I111=imnoise(img,'salt & pepper',0.025);
 I111=medfilt2(I111);
 K111=wiener2(I111);
 img=K111;
 img = imfilter(img,f2,'replicate');
    
%Feature Extraction
    
%    [ca ch cv cd]=dwt2(img,'haar');
%    [ca ch cv cd]=dwt2(ca,'haar');
   %[ca ch cv cd]=dwt2(ca,'haar');
   % [ca ch cv cd]=dwt2(ca ,'haar');
    pic=dct2(img);
    %pic=ca;  
    %pic_dct=reshape(pic(1:24,1:16),1,384);
%Triangular-shaped feature extraction
I=pic;
u=zeros(1,400);
radius=12;
z=zeros(radius*10);
i11=1;
while(i11<=radius)
    z(i11)=radius-i11+1;
    i11=i11+1;
end
length2=0;
j1=1;
for i11=1:radius
    q1=z(i11);
    u(1,j1:q1+j1-1)=I(i11,1:q1);
    length2=length2+q1;
    j1=j1+q1;
end
u=u(1,1:length2);    
    pic_dct=reshape(u,1,length2);

%-Feature Selection-%    
    
    opt=pic_dct.*GlobalBestP;
    
%Compute Euclidean Distance with each test vector

    d=zeros(1,trainimgs);
    
             for p=1:trainimgs
                 r = storeface{p};
     d(p) = sqrt(sum((r-opt).^2));    
             end 
             
     [val,index2]=min(d);                   %Minimum of Euclidean Distances
                                           %gives the Matched Vector
     
     if((ceil(index2/trainratio))==c)                %Increment Recognition
     rec=rec+1;   %Counter if successful
     end 
     end                                   %Recognition
  
end 
time2=toc;
 xlswrite('RR.xls', time2,'Target', sprintf('E%d',x));
disp('Recognition rate:');                 %Find Recognition Rate 
percent=(rec/tests)*100;
disp(percent);
xlswrite('RR.xls', percent,'Target', sprintf('C%d',x));
percentsum(x)=percent;

end
%displaying the results
disp('Average number of selected features:')%Find average number of
disp(sum(temp)/max(x));                     %selected features
xlswrite('RR.xls', percent,'Target', sprintf('C%d',x));
disp('Average Recognition Rate:')           %Find average of
disp(sum(percentsum)/max(x));               %Recognition rate  
     
