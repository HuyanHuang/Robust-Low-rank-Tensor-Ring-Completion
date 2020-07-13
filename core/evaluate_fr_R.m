function [FR,Em]=evaluate_fr_R(tnsr,P)
%% compute freedom ratio (hardness) of the completion problem
N=ndims(tnsr);
m=sum(P(:));
L=ceil(N/2);
J=size(tnsr);
df_m=zeros(L,1);
Em=zeros(L,1);
for n=1:L
    order=[n:N 1:n-1];
    X_temp=permute(tnsr,order);
    X=reshape(X_temp,prod(J(order(1:L))),[]);
    TRunfoldingr=rank(X);
    df_m(n)=TRunfoldingr*(sum(size(X))-TRunfoldingr)/m;
    Em(n)=max(size(X))*TRunfoldingr*log(max(size(X)))^2.5;
end
FR=mean(df_m);
end