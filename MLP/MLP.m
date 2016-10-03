classdef MLP < handle
    %MLP Class This class is the core of the implementation: it implements
    %the MLP algorithm as described in the theory. It inherits the handle
    %class to be able to make lasting edits of the variables 
    
    %% Properties of the class
    properties
        h1  % number of neurons in the first hidden level
        eta % learning rate
        mu  % momentum term
        W1 % weights of the first level (2h1 x d)
        w2 % weights of the second level (h1 x d)
        b1 % biases of the first level (2h1 x d)
        b2 % bias of the second level (scalar)
        dW1old % the old change of W1 (used because of the momentum)
        dw2old % analogue
        db1old % analogue
        db2old % analogue
        dataTr % the training set
        labelsTr % the training labels
        dataVal % the validation set
        labelsVal % the validation labels
        id % the ID of this MLP, this is to keep track of all the executions
        
    end
    
    methods
        %% Constuctor function
        function obj = MLP(h1, eta, mu, dataTr, labelsTr, dataVal, labelsVal,id)
            obj.h1 = h1;
            obj.eta = eta;
            obj.mu = mu;
            obj.dataTr = dataTr;
            obj.labelsTr = labelsTr;
            obj.dataVal = dataVal;
            obj.labelsVal = labelsVal;
            obj.id=id;
        end
        
        %% Run function
        function [convergenceEpoch] = run(obj)
            % The core function of this class. After initialisation this
            % function should be called to let the MLP run. Afterwards this
            % returns the epoch of convergence.
            n = size(obj.dataTr,1); % number of training patterns
            d = size(obj.dataTr,2); % dimension of the patterns 
            minErr = 10e10; % very high number
            % counter and delay are used to extend early stopping. If after
            % delay steps no lower validation error is found than the last
            % one (saved in minErr) we stop the algorithm
            counter = 0; 
            delay = 3; 
            
            sigma=1/sqrt(d); % to initialise the weights
            convergenceEpoch = 50; % initial value, used to indicate that no change has happened
            
            %initialisation of weights and other parameters
            obj.W1=random('norm',0,sigma,2*obj.h1,d);
            obj.w2=random('norm',0,sigma,obj.h1,1);
            obj.b1=random('norm',0,sigma,2*obj.h1,1);
            obj.b2=random('norm',0,sigma);
            obj.dW1old = zeros(2*obj.h1,d);
            obj.dw2old = zeros(obj.h1,1);
            obj.db1old = zeros(2*obj.h1,1);
            obj.db2old = 0;
            optW1 = obj.W1;
            optw2 = obj.w2;
            optb1 = obj.b1;
            optb2=obj.b2;
            
            % used for plotting and evaluating the MLP
            indices=zeros(1,1);
            trainErrors = obj.error_function(obj.dataTr,obj.labelsTr);
            validationErrors = obj.error_function(obj.dataVal,obj.labelsVal);
            zeroOneErrors = obj.zero_one_errors(obj.dataVal,obj.labelsVal);
            
            % to keep count of the number of iterations (epochs)
            j=0;
            while true
                
                perm = randperm(n); % to go in a random order through all the patterns
                if(j>5)
                    obj.eta = obj.eta*0.9; % the learning rate decreases over time
                end
                j = j + 1;
                for i=1:n % run over all patterns
                    sample = obj.dataTr(perm(i),:); % select a pattern
                    label = obj.labelsTr(perm(i)); % select its corresponing label
                    [a1,a2,z1] = obj.forward_pass(sample); % perform the forward pass as described
                    [r1,r2] = obj.backward_pass(label,a1,a2); % perform the backward pass as described
                    obj = obj.update_weights(sample,r1,r2,z1); % update the weights
                end
                
                errTr = obj.error_function(obj.dataTr,obj.labelsTr); % calculate all the errors, for plotting and early stopping
                errVal = obj.error_function(obj.dataVal,obj.labelsVal);
                zeroOne = obj.zero_one_errors(obj.dataVal,obj.labelsVal);
                indices=[indices j]; % this could go faster but as it happens at most 50 times in one complete run it is negligble
                trainErrors=[trainErrors errTr];
                validationErrors=[validationErrors errVal];
                zeroOneErrors = [zeroOneErrors zeroOne];
                
                if(convergenceEpoch==50) % didn't converge yet
                    if(errVal<minErr) % we found a lower validation error than we were tracking
                        counter = 0;
                        minErr = errVal;
                        optW1=obj.W1; % this is the optimal weight configuration
                        optw2=obj.w2;
                        optb1=obj.b1;
                        optb2=obj.b2;
                    else
                        if(counter == delay) % early stopping happens
                            convergenceEpoch = j-delay-1; 
                            obj.W1=optW1;
                            obj.w2=optw2;
                            obj.b1=optb1;
                            obj.b2=optb2;
% put the line below in comments in case you want to make a plot with indices from 0 to 50.                        
                            break; 
                        end
                        counter = counter + 1;
                    end
                end
                if(j==53) % we end the run, converged or not
                    obj.W1=optW1; % before leaving we restore the most optimal configuration
                    obj.w2=optw2;
                    obj.b1=optb1;
                    obj.b2=optb2;
                    break;
                end
                fprintf('|');
            end
            
            save('lastdata.mat','indices','trainErrors','validationErrors','zeroOneErrors'); % used for plotting
          
        end
        %% Function to compute the zero/one-error rate
        function ratio = zero_one_errors(obj,data,labels)
            [~,a2,~] = obj.forward_pass(data);
            prod = labels .* a2'; 
            corr = sum(prod < 0);
            ratio = corr/size(data,1);
        end
        %% Function to compute the forward pass (see MLP algorithm)
        function [a1,a2,z1] = forward_pass(obj,sample) 
            n = size(sample,1);
            B1 = obj.b1 * ones(1,n);
            a1 = obj.W1*sample'+B1;
            z1 = obj.gating(a1);
            a2 = obj.w2'*z1 + obj.b2;
        end
        %% Function to compute the backward pass (see MLP algorithm)
        function [r1,r2]=backward_pass(obj,label,a1,a2)
            r2=-label/(exp(label*a2)+1);
            a1odd = a1(1:2:2*obj.h1);
            a1even= a1(2:2:2*obj.h1);
            r1odd = r2 .* obj.w2 ./ (1+exp(-a1even));
            r1even= r2 .* obj.w2 .* a1odd ./ (exp(a1even)+2+exp(-a1even)); 
            r1 = zeros(2*obj.h1,1);
            r1(1:2:2*obj.h1,:) = r1odd;
            r1(2:2:2*obj.h1,:) = r1even;
        end     
        %% Function to compute the gradients and afterwards new weights (see MLP algorithm)
        function obj = update_weights(obj,sample,r1,r2,z1) 
            gradW1=r1*sample;   % the gradients 
            gradw2=r2.*z1;      
            gradb1=r1;          
            gradb2=r2;
            
            dW1 = -obj.eta*(1-obj.mu).*gradW1 + obj.mu.*obj.dW1old; % the change of the weights by using the momentum term
            obj.dW1old = dW1; % for the next computation of the new weights
            obj.W1 = obj.W1 + dW1; %update the weight
            
            dw2 = -obj.eta*(1-obj.mu).*gradw2 + obj.mu.*obj.dw2old;
            obj.dw2old = dw2;
            obj.w2 = obj.w2 + dw2;
            
            db1 = -obj.eta*(1-obj.mu).*gradb1 + obj.mu.*obj.db1old;
            obj.db1old = db1;
            obj.b1 = obj.b1 + db1;
            
            db2 = -obj.eta*(1-obj.mu).*gradb2 + obj.mu.*obj.db2old;
            obj.db2old = db2;
            obj.b2 = obj.b2 + db2;
                
        end
        %% Implementation of the gating function
        function z1 = gating(obj,a1)
            temp1 = ones(size(a1,1),size(a1,2)); 
            temp1(2:2:2*obj.h1,:)=0;
            temp2 = ones(size(a1,1),size(a1,2)) - temp1;
            temp1 = temp1 .* a1;
            temp2 = temp2 .* 1 ./ (1 + exp(-a1));
            temp1 = temp1(1:2:2*obj.h1,:);
            temp2 = temp2(2:2:2*obj.h1,:);
            z1 = temp1.*temp2;            
        end
        %% Implementation of the logistic error function (mean of the logistic errors)
        function err = error_function(obj,data,labels)
            [~,a2,~] = obj.forward_pass(data);  
            prod = -labels .* a2'; % (n x 1) = (n x 1) * (n x 1)
            pos = prod(prod>=0);
            neg = prod(prod<0);
            pos = pos + log(1 + exp(-pos));
            neg = log1p(exp(neg));
            res = [pos ; neg];
            err = mean(res);
        end
        %% Function to compute the standard deviation of the logistic errors
        function stddev = stddev_error(obj,data,labels)
            [~,a2,~] = obj.forward_pass(data);  
            prod = -labels .* a2'; 
            pos = prod(prod>=0);
            neg = prod(prod<0);
            pos = pos + log(1 + exp(-pos));
            neg = log1p(exp(neg));
            res = [pos ; neg];
            stddev = std(res);
        end

    end
    
end

