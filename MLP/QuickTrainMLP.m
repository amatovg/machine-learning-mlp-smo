%% useful function to do a quick MLP run
function QuickTrainMLP(h1,eta,mu)

    load('normData49.mat');
    [avgEpoch,avgValues]=TrainMLP(h1,eta,mu,1,Xtr,Ytr,Xva,Yva);
    fprintf('Average Convergence-epoch is %.3f and the average error-value is %.3f.\n',avgEpoch,avgValues);
end