%% function used for the report
function [largest,smallest] = indices_misclassified(instance,X,Y)
    [~,a2,~] = instance.forward_pass(X);  
    prod = Y .* a2'; % (n x 1) = (n x 1) * (n x 1)
    prod = prod(prod<0);
    [~, largest] = max(prod);
    [~, smallest] = min(prod);


end