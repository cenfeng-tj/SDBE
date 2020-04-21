function W = cal_W(A,B,lambda,USEGPU,MIXGPUCPU)
% A : CD
% B : OED
% W = A*Pa, where Pa is the CD portion of the projection matrix P, i.e., 
%           P = [Pa^T,Pb^T]^T. P is computed by
%           P = (D^T*D + lambda*I)^(-1)*D^T, where D = [A B].
% USEGPU : true: use GPU to accelerate computing
%          false: use CPU to compute W 
% MIXGPUCPU: Take effect when USEGPU==true
%            true: use CPU to compute the inverse of P
%            false: use GPU to compute the inverse of P  

D = double([A, B]);

if USEGPU
    D_g = gpuArray(D);
    D_g_T = gpuArray(D');
    P_g = D_g_T*D_g;
    I_g = lambda*gpuArray(eye(size(D,2)));
    clear D_g
    
    P_g = P_g + I_g;
    clear I_g
    
    if MIXGPUCPU
        InvP = inv(gather(P_g));
        clear P_g;
        InvP_g = gpuArray(InvP);
    else
        InvP_g = inv(P_g);
        clear P_g;
    end
    
    P_g = InvP_g*D_g_T;
    clear InvP_g D_g_T;
    
    W_g = gpuArray(double(A))*P_g(1:size(A,2),:);
    W = gather(W_g);
    
    clear P_g W_g;
else
    P = inv(D'*D+lambda*eye(size(D,2)))*D';
    W = double(A)*P(1:size(A,2),:);    
end






