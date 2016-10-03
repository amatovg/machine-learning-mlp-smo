% It apply here SMO on all training set with the best couple (C, tau) found
% with main script and give the classifier y (the SMO_withCriterionPlot is
% used for the plot).
% After it plot the criterion and violation.
% Then at the end it gives the error rate on training set and test set.

clear all

load('svm_data.mat')

dataSet = Xtr;
labels = Ytr;
C = 0.64; % best C found
tau = 0.096; % best tau found

[ violation, criterion, alpha, b ] = SMO_withCriterionPlot( C, tau, dataSet, labels );

% error on training set
K = Kernel( dataSet, tau );
y = ((alpha.*labels)'*K- b)';  % classifier on training set
err_onTrainingSet = length( find(sign(y)~=labels)) / length(labels);

% error on test set 
t = classifier_from_trainSet( alpha, b, dataSet, labels, tau, Xte ); % labels from classifier
err_onTestSet = length( find(sign(t)~=Yte)) / length(Yte);

iterations= 20*(1:length(criterion));

save('plotData.mat','iterations','criterion','violation','err_onTrainingSet','err_onTestSet', 'alpha', 'b');

load('plotData.mat');
figure;
subplot(1,2,1);
plot(iterations,criterion,'.-','Color','b');
title('Evolution of SVM Criterion','FontSize',12);
xlabel('SMO iterations','FontSize',12);                  
ylabel('SVM Criterion','FontSize',12); 
grid on;
subplot(1,2,2);
semilogy(iterations,violation,'.-','Color','r');
title('Evolution of Convergence Criterion','FontSize',12);
xlabel('SMO iterations','FontSize',12);                  
ylabel('KKT Condition Violations','FontSize',12); 
grid on;

fprintf([' (Zero/one) error on Training set = ' num2str(err_onTrainingSet) '\n ']);
fprintf([' (Zero/one) error on Test set = ' num2str(err_onTestSet) '\n ']);

% End 