% This function is modified from SolveDALM.m in l1benchmark.zip (http://people.eecs.berkeley.edu/~yang/software/l1benchmark/l1benchmark.zip)

function [x, nIter] = L1LS_DALM(AA, A, At, b, USEGPU, varargin)

STOPPING_INCREMENTS = 1 ;
STOPPING_DEFAULT = STOPPING_INCREMENTS;

stoppingCriterion = STOPPING_DEFAULT;

tol = 1e-3;
lambda = 1e-2;
maxIter = 5000;
VERBOSE = 0;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' lambdast be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'tolerance'
            tol = parameterValue;
        case 'lambda'
            lambda = parameterValue;
        case 'maxiteration'
            maxIter = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end
clear varargin

[m,n] = size(A);

beta = norm(b,1)/m;
betaInv = 1/beta ;

if USEGPU
    b = gpuArray(b);
end
G = AA + eye(m) * lambda / beta;
invG = inv(G);
A_invG_b = At * (invG * b);

nIter = 0 ;

if VERBOSE
    disp(['beta is: ' num2str(beta)]);
end

if USEGPU
    y = zeros(m,1,'gpuArray');
    x = zeros(n,1,'gpuArray');
    z = zeros(m+n,1,'gpuArray');
else
    y = zeros(m,1);
    x = zeros(n,1);
    z = zeros(m+n,1);
end

converged_main = 0 ;

temp = At * y;
f = norm(x,1);
while ~converged_main
    
    nIter = nIter + 1 ;  

    if USEGPU
        x_old = gather(x);
    else
        x_old = x;
    end
    
    %update z
    if USEGPU
        temp1CPU = gather(temp + x * betaInv);
    else
        temp1CPU = temp + x * betaInv;
    end
    z = sign(temp1CPU) .* min(1,abs(temp1CPU));
    
    %compute A' * y  
    temp = At * (invG * (A * (z - x * betaInv))) + A_invG_b * betaInv; 
    
    %update x
    x = x - beta * (z - temp);    
   
    if USEGPU
        xCPU = gather(x);
    else
        xCPU = x;
    end
    
    switch stoppingCriterion
        case STOPPING_INCREMENTS
              xCPU = gather(x);
              if norm(x_old - xCPU) < tol*norm(x_old)
                converged_main = 1 ;
            end
        otherwise
            error('Undefined stopping criterion.');
    end
      
    if ~converged_main && norm(x_old-xCPU)<100*eps
        if VERBOSE
            disp('The iteration is stuck.') ;
        end
        converged_main = 1 ;
    end
    
    if ~converged_main && nIter >= maxIter
        if VERBOSE
            disp('Maximum Iterations Reached') ;
        end
        converged_main = 1 ;
    end
    
end
if USEGPU
    x = gather(x);
end
