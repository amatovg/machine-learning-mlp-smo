function [ Xval, Xtra, Yval, Ytra ] = splitData( Xtrain,Ytrain )
%SPLITDATA Split the training part of the data into a training set and a
%validation set with 2/3 and 1/3 ratios
perm = randperm(size(Xtrain,1));
Xtra=zeros(size(Xtrain,1)*2/3,size(Xtrain,2));
Xval=zeros(size(Xtrain,1)*1/3,size(Xtrain,2));
Ytra=zeros(size(Ytrain,1)*2/3,1);
Yval=zeros(size(Ytrain,1)*1/3,1);
for i = 1:size(Xtrain,1)*2/3 % this is only done once, so for loops don't hurt here
    val = perm(i);
    Xtra(i,:)=Xtrain(val,:);
    Ytra(i,:)=Ytrain(val,:);
end
for i = 1:size(Xtrain,1)/3
    val = perm(size(Xtrain,1)*2/3+i);
    Xval(i,:)=Xtrain(val,:);
    Yval(i,:)=Ytrain(val,:);
end
end

