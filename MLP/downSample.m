%% function used for debugging; make the image smaller
function [ Xr ] = downSample( Xm, factor )
%DOWNSAMPLE Summary of this function goes here
%   Detailed explanation goes here
n = size(Xm,1);
d = size(Xm,2);
Xr = zeros(n,d/factor^2);
for i=1:n
	Xmat = reshape(Xm(i,:),sqrt(d),sqrt(d));
    Xmat = imresize(Xmat,1/factor);
    Xr(i,:) = reshape(Xmat,1,d/factor^2);
end

