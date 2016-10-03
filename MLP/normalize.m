function [ normTraSet,normValSet,normTestSet ] = normalize( traSet,valSet,testSet )
%NORMALIZE Normalize all data
maximum=max(max(max(traSet)),max(max(valSet)));
minimum=min(min(min(traSet)),min(min(valSet)));

normTraSet=(traSet-ones(size(traSet,1),size(traSet,2))*minimum)./(maximum-minimum);
normValSet=(valSet-ones(size(valSet,1),size(valSet,2))*minimum)./(maximum-minimum);
normTestSet=(testSet-ones(size(testSet,1),size(testSet,2))*minimum)./(maximum-minimum);



end

