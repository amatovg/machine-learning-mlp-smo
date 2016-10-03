%% Function used to display a data point as an image
function [] = dataToImage( Xmatrix, row )
%DATATOIMAGE Xmatrix is the matrix containing N different images in grayscale. row is the image you want to display (the row in Xmatrix)
Xrow=Xmatrix(row,:);
Xmat = reshape(Xrow,sqrt(size(Xrow,2)),sqrt(size(Xrow,2)));
imshow(mat2gray(Xmat))

end

