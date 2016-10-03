% function Kernel
%   - inputs : tau, and trainingSet x
%   - Output : K the kernel matrix with : for all i, j in [1, length(x)], Kij = exp(-0.5*tau*|xi-xj|^2)

function [ K ] = Kernel( x, tau )
d=sum(x.*x,2); % the diagonal of x*x'
A = 0.5*d*ones(1,length(d)) + 0.5*ones(length(d),1)*d'-x*x';
K = exp(-tau*A);
end

