% function classifier_from_trainSet
%   - inputs : alpha, b (given by SMO), trainingSet, label_Of_TrainingSet, tau and a validationSet (or Test set).
%   - Output : y(testSet)
%   The trick here is to compute the 'Kernel' of both training set and
%   validation set, knoxing that : 
%   for all (i, j) in [1, length(XtrainSet)]x[1, length(x)] , Kij = exp(-0.5*tau*|XtrainSet_i-x_j|^2)

function [ y ] = classifier_from_trainSet( alpha, b, XtrainSet, Labels_trainSet, tau, x )
intMat1 = XtrainSet*XtrainSet';
intMat2 = x*x';
d1 = diag(intMat1);
d2 = diag(intMat2);
n1 = length(XtrainSet(:,1)); % n1 = 5400
n2 = length(x(:,1)); % n2 = 600

A = d1*ones(1,n2) + ones(n1,1)*d2' - 2*XtrainSet*x';
K = exp(-(1/2)*tau*A);

y = ((alpha.*Labels_trainSet)'*K- b)';
end

