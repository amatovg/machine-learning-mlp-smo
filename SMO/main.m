%% The Main Script
% it is the main script so as to determine the best couple (C, tau),
% it goes through 100 couples (C, tau) with 10 differents C and 10 differents tau. 
% Knowing that we will use 10 fold cross-validation set, 
% the first step is to have 10 sub_trainSet and 10 sub_validation set from
% the TrainSet with each train set taking 9/10 of training data set and the validation set taking the 10 last per cent.
% We then pick a couple (C, tau), we run the SMO on each training Set 
% and we compute the error on the corresponding validation set, and compute the mean error and record it in a matrix.
% At the end we have a matrix of error according to the couple C, tau. 
% We record the best couple (C, tau). 


clear all

%% load splitted from normalizedData (10 set for cross validation)

load('svm_data.mat')
Xval = cell(1,10);
Yval = cell(1,10);
XtrN= cell(1,10);
YtrN = cell(1,10);
for i=1:10 
    Xval{i} = Xtr(1+length(Ytr)/10*(i-1) : length(Ytr)/10+length(Ytr)/10*(i-1), :);
    Yval{i} = Ytr(1+length(Ytr)/10*(i-1) : length(Ytr)/10+length(Ytr)/10*(i-1), :);
    XtrN{i} = [];
    YtrN{i} = [];
end

for i=1:10 
    for j=1:10
        if i~=j
            XtrN{i} = [XtrN{i}; Xval{j}];
            YtrN{i} = [YtrN{i}; Yval{j}];
        end
    end
end

%% setup the value of C and tau which will be tested
p_ow = [0 1 2 3 4 5 6 7 8 9];
C_val = 0.00125.*2.^p_ow;
tau_val = 0.0015.*2.^p_ow;

error_best = 1e10;

fid = fopen('result.txt','a');  %we also record results on a txt file
time = strcat('\n\n\n',date,'_',datestr(now, 'HH:MM:SS'),'\n\n\n');
fprintf(fid,time);

ERROR_MAT = zeros(length(tau_val)+1, length(C_val)+1);

for k=1:length(C_val)
    C = C_val(k);
    ERROR_MAT(1,k+1)=C;
    
   for l=1:length(tau_val)
       tau = tau_val(l);
       ERROR_MAT(l+1,1)=tau;
       err = 0; % error initialization before computed it for a each new couple (C, tau)
       
       for i=1:10
            dataSet = XtrN{i};
            labels = YtrN{i};
            [alpha, b] = SMO( C, tau, dataSet, labels );  %classifier train on subTrainset i
            %compute error validationSet_i with "alpha_i and b_i"
            y = classifier_from_trainSet( alpha, b, dataSet, labels, tau, Xval{i} ); % labels from classifier
            err = err + length( find(sign(y)~=Yval{i})) / length(Yval{i});
            fprintf('%d',i);
       end
       err = err/10;
       ERROR_MAT(l+1,k+1)=err;
       %write the mean error with (C, tau)
       fprintf(fid,[' Classification error based on C = ' num2str(C) ' and tau = ' num2str(tau) ' is ' num2str(err) '\n ']);
       fprintf([' Classification error based on C = ' num2str(C) ' and tau = ' num2str(tau) ' is ' num2str(err) '\n ']);   
       if err <= error_best
           C_best = C;
           tau_best = tau;
           error_best = err;
       end
   end
end
fprintf(fid,['best C is = ' num2str(C_best) '\n' ]);
fprintf(fid,['best tau is = ' num2str(tau_best) '\n' ]);
fprintf(fid,['best error is = ' num2str(error_best) '\n' ]);
fprintf(['best C is = ' num2str(C_best) '\n' ]);
fprintf(['best tau is = ' num2str(tau_best) '\n' ]);
fprintf(['best error is = ' num2str(error_best) '\n' ]);

fclose(fid);

save('error_mat.mat','ERROR_MAT');