function [M1,num1]=shrink_matrix(M0,tau,num0,epsilon,flag)
[m,n]=size(M0);
if flag
    if m<1.5*n
        [U,Sigma2]=eigs(M0*M0',num0,'largestreal','MaxIterations',500);
        sigma=sqrt(diag(Sigma2));
        s=sigma-tau;
        idx_s=find(s>0,1,'last');
        num1=find(s>=epsilon*s(1),1,'last');
        S_shrink=diag(s(1:idx_s)./sigma(1:idx_s));
        M1=U(:,1:idx_s)*S_shrink*U(:,1:idx_s)'*M0;
        return
    end
    if m>1.5*n
        [M1,num1]=shrink_matrix(M0',tau,num0,epsilon,flag);
        M1=M1';
        return
    end
    [U,S,V]=svds(M0,num0,'largest','MaxIterations',500);
    s=diag(S)-tau;
    idx_s=find(s>0,1,'last');
    num1=find(s>=epsilon*s(1),1,'last');
    S_shrink=diag(s(1:idx_s));
    M1=U(:,1:idx_s)*S_shrink*V(:,1:idx_s)';
else
    [U,S,V]=svd(M0,'econ');
    s=diag(S);
    idx=find(s>tau,1,'last');
    M1=U(:,1:idx)*diag(s(1:idx)-tau)*V(:,1:idx)';
    num1=num0;
end