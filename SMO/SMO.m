% function SMO
%   - inputs : C, tau, trainingSet, label_Of_TrainingSet
%   - Output : alpha, b
%   -  the sequential minimal optimization (SMO) algorithm to solve the soft margin support vector machine (SVM)
%   binary classification problem. SMO gives us our classifier given the data, labels and the parameters
%   C and tau fixed.

function [ alpha, b ] = SMO( C, tau, dataSet, labels ) 
    %SMO Summary of this function goes here
    %   Detailed explanation goes here

    alpha = zeros(length(labels), 1);
    K = Kernel(dataSet, tau);
    f = - labels;
    % compute index sets I_low and I_up
    Io = [];  % initially empty as all alpha's are zero
    I_plus = union(intersect(find (labels == +1), find( alpha == 0)), intersect(find (labels == -1), find( alpha == C))); 
    I_minus = union(intersect(find (labels == -1), find( alpha == 0)), intersect(find (labels == +1), find( alpha == C)));
    I_up = union ( Io, I_plus);
    I_low = union ( Io, I_minus);
    
    count = 0;
    while(1)
       % procedure select pair 
       % Compute i_up, i_low
        i_up_all = I_up(f(I_up)==min(f(I_up)));   %i_up = argmin (fi such that i in I_up)
        i_low_all = I_low(f(I_low)==max(f(I_low)));   %i_low = argmax (fi such that i in I_low)
        i_up = i_up_all(1);   %apparently according to the TA, it does not matter wich i_up we take
        i_low = i_low_all(1);   % //
        % Check for optimality
        b = (f(i_low) + f(i_up))/2;
        if (f(i_low) <= f(i_up) + 2* tau)  %f_i_low <= f_i_up + 2* tau
            i_low = -1;
            i_up = -1;
        end
        %w = alpha(i) + sigma*alpha(j);
        i = i_low;
        j = i_up;
        % stop if the conditions are fulfilled f(i_low) <= f(i_up) + 2* tau
        
        if (j==-1)
            break;
        end

        sigma = labels(i)*labels(j);
        % compute L, H
        w = alpha(i) + sigma*alpha(j); 
        L = max ([0, sigma*w - (sigma == +1)*C]);
        H = min ([C, sigma*w + (sigma == -1)*C]);
        eta = K(i,i) + K(j, j) - 2* K(i, j);
        
        
        if (eta > 10^(-15))
            alpha_j_unc = alpha(j) + labels(j)*(f(i)-f(j))/eta;
            if L <= alpha_j_unc && alpha_j_unc <= H
                alpha_new_j = alpha_j_unc;   
            elseif alpha_j_unc < L
                alpha_new_j = L;
            else 
                alpha_new_j = H;
            end
        else
            % compute phi_H, phi_L according to (8)
            % compute v_i, v_j before update i, j. To be used later when computing
            v_i = f(i) + labels(i) - alpha(i)*labels(i)*K(i,i) - alpha(j)*labels(j)*K(i,j);
            v_j = f(j) + labels(j) - alpha(i)*labels(i)*K(i,j) - alpha(j)*labels(j)*K(j,j);
            L_i = w - sigma*L;
            phi_L = 0.5 * (K(i,i)*L_i^2 + K(j,j)*L^2) + sigma*K(i,j)*L_i*L + labels(i)*L_i*v_i + labels(j)*L*v_j - L_i - L;
            H_i = w - sigma*H;
            phi_H = 0.5 * (K(i,i)*H_i^2 + K(j,j)*H^2) + sigma*K(i,j)*H_i*H + labels(i)*H_i*v_i + labels(j)*H*v_j - H_i - H;

            if (phi_L > phi_H)
                alpha_new_j = H;
            else
                alpha_new_j = L;
            end    
        end

        alpha_new_i = alpha(i) + sigma*(alpha(j) - alpha_new_j);

        f = f + labels(i)*(alpha_new_i - alpha(i))*K(:, i) + labels(j)*(alpha_new_j - alpha(j))*K(:, j);

        % update alpha vector
        alpha(i) = alpha_new_i;
        alpha(j) = alpha_new_j;

        %update I_low and I_up
        Io = intersect(find(0 < alpha),find(alpha < C));  
        I_plus = union(intersect(find (labels == +1), find( alpha == 0)), intersect(find (labels == -1), find( alpha == C)));
        I_minus = union(intersect(find (labels == -1), find( alpha == 0)), intersect(find (labels == +1), find( alpha == C)));
        I_up = union ( Io, I_plus);
        I_low = union ( Io, I_minus);
        if(mod(count,100)==0)
            fprintf('|');
        end
        if(count > 3000)    % against endless loops
            break;
        end
        count = count + 1;
    end
    fprintf('\n');
end

