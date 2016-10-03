%% Function to train the MLP: initialise it an let it run with the given parameters and data
function [avgEpoch,avgVal,avgTra,instance]=TrainMLP(h1,eta,mu,logID,Xtr,Ytr,Xva,Yva)
    % h1, eta, mu are the parameters
    % logID is used for the logging
    % all X's and Y's are the data
    % avgEpoch: average convergence epoch of all runs performed
    % avgVal: average validation error of all runs performed
    % avgTra: average training error of all runs performed
    % instance: an object of class MLP; can be used for calculating error
    %   functions for example
    
    opened = 0;
    if(logID==-1) % open new file
        logID=fopen('MLP_evaluation_logs.txt','r+');
        opened = 1;
    end
    load('ID.mat'); % to keep track of all different runs
    number=5; 
    % number is the number of returned results over which the average will 
    % be computedof all runs. This is to reduce the effect of the noise
    % on the final results.
    fprintf(logID,'Run with h1 = %d, eta = %g, mu = %g\n',h1,eta,mu);
    epochs=zeros(1,number);
    valValues=zeros(1,number);
    traValues=zeros(1,number);
    
    for i=1:number
        instance = MLP(h1,eta,mu,Xtr,Ytr,Xva,Yva,ID); % initialise the MLP algorithm
        ID=ID+1; 
        epochs(i) = instance.run(); % run the algorithm
        fprintf('.');
        fprintf(logID,'Convergence Epoch is %d after %.2fs\n',epochs(i),toc);
        valValues(i) = instance.error_function(instance.dataVal,instance.labelsVal);
        traValues(i) = instance.error_function(instance.dataTr,instance.labelsTr);
        fprintf('\nThe logistic error on the trainset is %.3f.',errTr);
        fprintf(logID,'The logistic error on the validationset is %.3f.\n\n',values(i));
    end
    fprintf('\n');
    avgEpoch=mean(epochs);
    avgVal = mean(valValues);
    avgTra = mean(traValues);
    save('ID.mat','ID');
    if(opened == 1)
        fclose(logID);
    end
end
