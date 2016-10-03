% What happens here is the generation of all the data for the plots in our
% report for the MLP Evaluation. We define a search space of parameters and
% while we keep two parameters constant we wish to study the effect of the
% free parameter on the convergence and error. 
% In the end we do an
% exhaustive search over all the parameters in order to find the best
% combination. As this takes a lot of time commenting out might be a good
% idea.
% All of our results and logs are printed to two designated text files
% which can be used in order to find earlier computed values.

clear;
valuesID=fopen('MLP_evaluation_values.txt','a');
logID=fopen('MLP_evaluation_logs.txt','a');

time = strcat('\n\n\n',date,'_',datestr(now, 'HH:MM:SS'),'\n\n\n');

fprintf(logID,time);
fprintf(valuesID,time);
fprintf(valuesID,'h1\teta\tmu\tconv\tvalerr\ttraerr\n');

h1=[5 10 20 40 60 80 100];
eta=[0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.2 0.25 0.5];
mu=[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

h1_35_epochs = zeros(1,size(h1,2));
h1_35_values = zeros(1,size(h1,2));
h1_35_tr_values = zeros(1,size(h1,2));


load('normData35.mat');
for i=1:size(h1,2)
    for j=5:5 % eta = 0.8
        for k=3:3 % mu = 0.2
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            h1_35_epochs(i) = avgEpoch;
            h1_35_values(i) = avgVal;
            h1_35_tr_values(i) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('h1_35.mat','h1','h1_35_epochs','h1_35_values','h1_35_tr_values')

h1_49_epochs = zeros(1,size(h1,2));
h1_49_values = zeros(1,size(h1,2));
h1_49_tr_values = zeros(1,size(h1,2));

load('normData49.mat');
for i=1:size(h1,2)
    for j=5:5 % eta = 0.8
        for k=3:3 % mu = 0.2
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            h1_49_epochs(i) = avgEpoch;
            h1_49_values(i) = avgVal;
            h1_49_tr_values(i) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('h1_49.mat','h1','h1_49_epochs','h1_49_values','h1_49_tr_values')

eta_35_epochs = zeros(1,size(eta,2));
eta_35_values = zeros(1,size(eta,2));
eta_35_tr_values = zeros(1,size(eta,2));

load('normData35.mat');
for i=2:2 %h1 = 10
    for j=1:size(eta,2)
        for k=3:3 % mu = 0.2
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            eta_35_epochs(j) = avgEpoch;
            eta_35_values(j) = avgVal;
            eta_35_tr_values(j) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('eta_35.mat','eta','eta_35_epochs','eta_35_values','eta_35_tr_values')

eta_49_epochs = zeros(1,size(eta,2));
eta_49_values = zeros(1,size(eta,2));
eta_49_tr_values = zeros(1,size(eta,2));

load('normData49.mat');
for i=2:2 %h1 = 10
    for j=1:size(eta,2)
        for k=3:3 % mu = 0.2
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            eta_49_epochs(j) = avgEpoch;
            eta_49_values(j) = avgVal;
            eta_49_tr_values(j) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('eta_49.mat','eta','eta_49_epochs','eta_49_values','eta_49_tr_values')

mu_35_epochs = zeros(1,size(mu,2));
mu_35_values = zeros(1,size(mu,2));
mu_35_tr_values = zeros(1,size(mu,2));

load('normData35.mat');
for i=2:2 %h1 = 10
    for j=5:5 % eta = 0.08
        for k=1:size(mu,2)
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            mu_35_epochs(k) = avgEpoch;
            mu_35_values(k) = avgVal;
            mu_35_tr_values(k) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('mu_35.mat','mu','mu_35_epochs','mu_35_values','mu_35_tr_values')

mu_49_epochs = zeros(1,size(mu,2));
mu_49_values = zeros(1,size(mu,2));
mu_49_tr_values = zeros(1,size(mu,2));

load('normData49.mat');
for i=2:2 %h1 = 10
    for j=5:5 % eta = 0.08
        for k=1:size(mu,2)
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            mu_49_epochs(k) = avgEpoch;
            mu_49_values(k) = avgVal;
            mu_49_tr_values(k) = avgTra;
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end
save('mu_49.mat','mu','mu_49_epochs','mu_49_values','mu_49_tr_values')

%% Exhaustive Search for optimal parameters
% This merely prints all the results to a text file, further research needs
% to be done by importing the file into excel.

load('normData49.mat');

for i=1:length(h1)-2 % high eta and h1 values never seem to lead to the combination of a good classifier and fast convergence.
    for j=1:length(eta)-2
        for k=1:length(mu)
            v1=h1(i);v2=eta(j);v3=mu(k);
            fprintf(strcat('h1__',num2str(v1),'__eta__',num2str(v2),'__mu__',num2str(v3),'\n'));
            [avgEpoch,avgVal,avgTra]=TrainMLP(v1,v2,v3,logID,Xtr,Ytr,Xva,Yva);
            text=strcat(num2str(v1),'\t',num2str(v2),'\t',num2str(v3),'\t',num2str(avgEpoch,2),'\t',num2str(avgVal,3),'\t',num2str(avgTra,3),'\n');
            fprintf('Average Convergence-epoch is %.3f and the average val-error-value is %.3f and the average tra-error-value is %.3f.\n',avgEpoch,avgVal,avgTra);
            fprintf(valuesID,text);
        end
    end
end

%% Close all open files

fclose('all');