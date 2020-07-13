function [x,y,RC,run_time]=RTRC(tnsr,P,mu,flag)
%% initialize parameters
N=ndims(tnsr);
J=size(tnsr);
x=P.*tnsr;
x0=x;
y=zeros(J);
y0=y;
L=ceil(N/2);
l=cell(L,1);
z=cell(L,1);
w=zeros(J);
lambda=0;
sk=zeros(L,1);
SR=sum(P(:))/numel(P);
for n=1:L
    l{n}=x;
    z{n}=zeros(J);
    order=[n:N 1:n-1];
    lambda=lambda+1/sqrt(SR*max([prod(J(order(1:L))),prod(J(order(L+1:N)))]));
    sk(n)=min([prod(J(order(1:L))),prod(J(order(L+1:N)))]);
end
maxiter=100;
epsilon_x=1e-8;
epsilon_y=1e-8;
l_cs=zeros(J);
z_cs=zeros(J);
RC=nan(maxiter,2);
%% ADMM algorithm
[FR,Em]=evaluate_fr_R(x,P);
weight=1./Em;
weight=weight/sum(weight);
if FR>=3
    epsilon=1e-1;
elseif FR>=2
    epsilon=1e-2;
else
    epsilon=1e-3;
end
t=cputime;
for i=1:maxiter
    % update l^(n) (auxiliary variables of low-rank part)
    for n=1:L
        order=[n:N 1:n-1];
        m=permute(x-z{n}/mu,order);
        M=reshape(m,prod(J(order(1:L))),[]);
        [M,sk(n)]=shrink_matrix(M,weight(n)/mu,sk(n),epsilon,false);
        m=reshape(M,J(order));
        l{n}=ipermute(m,order);
        l_cs=l_cs+l{n};
        z_cs=z_cs+z{n};
    end
    % update x (low-rank part)
    x=(l_cs+z_cs/mu+P.*(tnsr-y-w/mu))./(L*ones(J)+P);
    l_cs=zeros(J);
    z_cs=zeros(J);
    % update y (sparse part)
    y=shrink_vector(P.*(tnsr-x-w/mu),lambda/mu);
    % update z^(n)
    for n=1:L
        z{n}=z{n}+mu*(l{n}-x);
    end
    % update w
    w=w+mu*P.*(x+y-tnsr);
    % evaluate recovery accuracy
    RC(i,:)=[norm(x(:)-x0(:),2)/norm(x0(:),2),norm(y(:)-y0(:),2)/norm(y0(:),2)];
    if flag
        fprintf('Iteratoin=%d\tRC=%f, %f\n',i,RC(i,1),RC(i,2));
    end
    if RC(i,1)<epsilon_x && RC(i,2)<epsilon_y
        break
    end
    x0=x;
    y0=y;
    mu=min(mu*1.1,1e10);
end
run_time=cputime-t;
fprintf('running time=%fs\n',run_time);
end