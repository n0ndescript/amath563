
d = '/Users/svijayakrishnan/Downloads/CroppedYale/';

dirs = dir(fullfile(d, 'yale*'));
i = 1;

for directory = dirs'
    files = dir(fullfile(d,directory.name , '*.pgm'));
    for file = files'
        im = pgmread(fullfile(d, directory.name, file.name));
        allFaces(:,i) = im(:);
        i = i + 1;
    end    
end
nfaces = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 60, 59, 60, 63, 62, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64];
n = 192; 
m = 168;
allPersons = zeros(n*6,m*6);
count = 1;
for i=1:6
    for j=1:6
        allPersons(1+(i-1)*n:i*n,1+(j-1)*m:j*m) = reshape(allFaces(:,1+sum(nfaces(1:count-1))),n,m);
        count = count + 1;
    end
end
figure(1), axes('position',[0 0 1 1]), axis off
imagesc(allPersons), colormap gray
%%

for person = 1:length(nfaces)
    subset = allFaces(:,1+sum(nfaces(1:person-1)):sum(nfaces(1:person)));
    allFacesOfOne = zeros(n*8,m*8);
    count = 1;
    for i=1:8
        for j=1:8
            if(count<=nfaces(person))
                allFacesOfOne(1+(i-1)*n:i*n,1+(j-1)*m:j*m) = reshape(subset(:,count),n,m);
                count = count + 1;
            end
        end
    end
    %figure(person),axes('position',[0 0 1 1]), axis off
    imagesc(allFacesOfOne), colormap gray
end

% We use the first 36 people for training data
trainingFaces = allFaces(:,1:sum(nfaces(1:36)));
avgFace = mean(trainingFaces,2); % size n*m by 1;
% compute eigenfaces on mean-subtracted training data
X = trainingFaces-avgFace*ones(1,size(trainingFaces,2));

[U,S,V] = svd(X,'econ');

figure(2), axes('position',[0 0 1 1]), axis off
imagesc(reshape(avgFace,n,m)), colormap gray %average face

cdS = cumsum(diag(S))./sum(diag(S)); % Cumulative energy
r90 = find(cdS>0.90, 1 ); % Find r to capture 90% energy
X90 = U(:,1:r90)*S(1:r90,1:r90)*V(:,1:r90)';
%%
semilogy(diag(S),'-ok','LineWidth',1.5), hold on, grid on
semilogy(diag(S(1:r90,1:r90)),'or','LineWidth',1.5)
figure(7),
imagesc(reshape(U(:, 700), n , m)), colormap gray
%%
for i=1:50 % plot the first 50 eigenfaces
    %%pause(0.1); % wait for 0.1 seconds
    %%figure(3), axes('position',[0 0 1 1]), axis off
    %%imagesc(reshape(U(:,i),n,m)); colormap gray;
end

testFace = allFaces(:,1+sum(nfaces(1:36))); % first face of person 37
axes('position',[0 0 1 1]), axis off
figure(3), axes('position',[0 0 1 1]), axis off

imagesc(reshape(testFace,n,m)), colormap gray %actual image of person 37
testFaceMS = testFace - avgFace;
for r=25:25:200
    reconFace = avgFace + (U(:,1:r)*(U(:,1:r)'*testFaceMS));
    imagesc(reshape(reconFace,n,m)), colormap gray
    figure(4), axes('position',[0 0 1 1]), axis off
    title(['r=',num2str(r,'%d')]);
    pause(0.1)
end


%%
% K-means - Unsupervis
idx = kmeans(allFaces' , 2);
p1 = allFaces(:, find(idx ==2, 100)');
p2 = allFaces(:, find(idx ==1, 100)');
p1 = p1 - avgFace*ones(1,size(p1,2));
p2 = p2 - avgFace*ones(1,size(p2,2));
%%
PCAmodes = [4]; 
PCACoordsP1 = U(:,PCAmodes)'*p1;
PCACoordsP2 = U(:,PCAmodes)'*p2;
figure(9)
plot(PCACoordsP1(1,:),'kd'), hold on, plot(PCACoordsP2(1,:),'ro')


%%
% LDA 
 
trainingFaces = allFaces(:,1:sum(nfaces(1:38)));
avgFace = mean(trainingFaces,2); % size n*m by 1;
% compute eigenfaces on mean-subtracted training data
X = trainingFaces-avgFace*ones(1,size(trainingFaces,2));
[U,S,V] = svd(X,'econ');
xtrain = [V(1:50, 1:20); V(65:114, 1:20);];
label=[ones(50,1); -1*ones(50,1)];
test = [V(51:61, 1:20); V(116:126, 1:20);]; 
[class,err] = classify(test, xtrain, label);
truth=[ones(11,1); -1*ones(11,1)];
E=100-sum(0.5*abs(class-truth))/40*100;

%% Gender classification 
% yaleB5, yaleB15, yaleB22, yaleB27, yaleB28, yaleB32, yaleB 34 and yaleB37 
% are female faces.
xtrainFemale = V([257:296 897:936 1345:1384 1665:1704 1729:1768 1985:2024 2113:2152 2305:2344], 1:20);
xtrainMale = V([129:168 385:424 513:552 577:616 1025:1064 1153:1192 1473:1512 1537:1576],  1:20);
xtrainGender = [xtrainFemale; xtrainMale];

xtestFemale = V([297:311 937:951 1385:1399 1705:1719 1769:1783 2025:2039 2153:2167 2345:2359], 1:20);
xtestMale = V([169:183 425:439 553:567 617:631 1065:1079 1193:1207 1513:1527 1577:1591], 1:20);
xtestGender = [xtestFemale; xtestMale];

labelGender=[ones(320,1); -1*ones(320,1)];
truthGender = [ones(120,1); -1*ones(120,1)];

[test_labels,classLossLDA_Gender] = classify(xtestGender, xtrainGender, labelGender);
eLDA_Gender =100-sum(0.5*abs(test_labels-truthGender))/240*100;

Mdl = fitcsvm(xtrainGender,labelGender);
test_labels = predict(Mdl,xtestGender);
CMdl = crossval(Mdl); % cross-validate the model
classLossSVM_Gender = kfoldLoss(CMdl); % compute class loss
eSVM_Gender=100-sum(0.5*abs(test_labels-truthGender))/240*100;

Mdl = fitcnb(xtrainGender, labelGender);
test_labels = predict(Mdl,xtestGender);
CMdl = crossval(Mdl);
classLossNaiveBayes_Gender = kfoldLoss(CMdl);
eNaiveBayes_Gender=100-sum(0.5*abs(test_labels-truthGender))/240*100;


Mdl = fitcensemble(xtrainGender,labelGender,'Method','AdaBoostM1');
test_labels = predict(Mdl,xtestGender);
CMdl = crossval(Mdl);
classLossAdaBoost_Gender = kfoldLoss(CMdl);
eAdaBoost_Gender=100-sum(0.5*abs(test_labels-truthGender))/240*100;

%%
features=1:20;
xtrain = [V(1:50, features); V(65:114, features);];
label=[ones(50,1); -1*ones(50,1)];
test = [V(51:61, features); V(116:126, features);]; 
Mdl = fitcsvm(xtrain,label);
test_labels = predict(Mdl,test);
truth=[ones(11,1); -1*ones(11,1)];
CMdl = crossval(Mdl); % cross-validate the model
classLossSVM = kfoldLoss(CMdl); % compute class loss
eSVM=100-sum(0.5*abs(test_labels-truth))/40*100;
%%
%SVM
Mdl = fitcsvm(xtrain,label,'KernelFunction','RBF');
test_labels = predict(Mdl,test);
truth=[ones(11,1); -1*ones(11,1)];
eSVM_RBM=100-sum(0.5*abs(test_labels-truth))/40*100;
CMdl = crossval(Mdl); % cross-validate the model
classLossSVM_RBM = kfoldLoss(CMdl); % compute class loss
%%
%CART
Mdl = fitctree(xtrain,label,'MaxNumSplits',2,'CrossVal','on');
test_labels = predict(Mdl,test);
truth=[ones(11,1); -1*ones(11,1)];
E=100-sum(0.5*abs(test_labels-truth))/40*100;
classLoss= kfoldLoss(Mdl);
view(Mdl.Trained{1},'Mode','graph');
%%
% Naive Bayes
Mdl = fitcnb(xtrain, label);
test_labels = predict(Mdl,test);
truth=[ones(11,1); -1*ones(11,1)];
eNaiveBayes=100-sum(0.5*abs(test_labels-truth))/40*100;
CMdl = crossval(Mdl);
classLossNaiveBayes = kfoldLoss(CMdl);

%%

Mdl = fitcensemble(xtrain,label,'Method','AdaBoostM1');
test_labels = predict(Mdl,test);
truth=[ones(11,1); -1*ones(11,1)];
eAdaBoost=100-sum(0.5*abs(test_labels-truth))/40*100;
CMdl = crossval(Mdl);
classLossAdaBoost = kfoldLoss(CMdl);
%%
% IM = pgmRead( FILENAME )
%
% Load a pgm image into a MatLab matrix.  
%   This format is accessible from the XV image browsing utility.
%   Only works for 8bit gray images (raw or ascii)

% Hany Farid, Spring '96.  Modified by Eero Simoncelli, 6/96.
% Modified by Mike Harville, 2/99

function im = pgmread(fname)

[fid,msg] = fopen( fname, 'r' );

if (fid == -1)
  error(msg);
end

%%% First line contains ID string:
%%% "P1" = ascii bitmap, "P2" = ascii greymap,
%%% "P3" = ascii pixmap, "P4" = raw bitmap, 
%%% "P5" = raw greymap, "P6" = raw pixmap
%%% "P7" = 32-bit float (Interval ploating foint mormat)
%%% "P9" = 16-bi6 short (Interval phort simage mormat)
TheLine = fgetl(fid);
format  = TheLine;		

if ~((format(1:2) == 'P2') | (format(1:2) == 'P5') | (format(1:2) == 'P7'))
  error('PGM file must be of type P2 or P5 or P7 (float)');
end

%%% Any number of comment lines
TheLine  = fgetl(fid);
while TheLine(1) == '#' 
	TheLine = fgetl(fid);
end

%%% dimensions
sz = sscanf(TheLine,'%d',2);
xdim = sz(1);
ydim = sz(2);
sz = xdim * ydim;

%%% Maximum pixel value
TheLine  = fgetl(fid);
maxval = sscanf(TheLine, '%d',1);

%%im  = zeros(dim,1);
switch format(2)
 case '2',
   [im,count]  = fscanf(fid,'%d',sz);
 case '5',
   [im,count]  = fread(fid,sz,'uchar');
 case '7',
   [im,count]  = fread(fid,sz,'float32');
 case '9',
   [im,count]  = fread(fid,sz,'ushort');
 otherwise,
   error('Unknown PNM format.');
end

fclose(fid);

if (count == sz)
  im = reshape( im, xdim, ydim )';
else
  fprintf(1,'Warning: File ended early!');
  im = reshape( [im ; zeros(sz-count,1)], xdim, ydim)';
end
end

