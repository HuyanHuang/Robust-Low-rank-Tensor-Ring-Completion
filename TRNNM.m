function [x,RC,RE,run_time]=TRNNM(tnsr,P,mu,flag)
%% initialize parameters
maxiter=100;
epsilon_x=1e-5;
N=ndims(tnsr);
J=size(tnsr);
x=P.*tnsr;
x0=x;
L=floor(N/2);
m=cell(N,1);
y=cell(N,1);
sk=zeros(N,1);
for n=1:N
    m{n}=x;
    y{n}=zeros(J);
    order=[n:N 1:n-1];
    sk(n)=min([prod(J(order(1:L))),prod(J(order(L+1:N)))]);
end
m_cs=zeros(J);
y_cs=zeros(J);
idx_o=logical(P);
RC=nan(maxiter,1);
RE=nan(maxiter,1);
%% ADMM algorithm
w=ones(1,N)/N;
t=cputime;
% main loop
for i=1:maxiter
    % update m^(n)
    for n=1:N
        order=[n:N 1:n-1];
        Z_temp=permute(x-y{n}/mu,order);
        Z=reshape(Z_temp,prod(J(order(1:L))),[]);
        [M,sk(n)]=shrink_matrix(Z,w(n)/mu,sk(n),[],false);
        M_temp=reshape(M,J(order));
        m{n}=ipermute(M_temp,order);
        m_cs=m_cs+m{n};
        y_cs=y_cs+y{n};
    end
    % update x
    x=(m_cs+y_cs/mu)/N;
    x(idx_o)=tnsr(idx_o);
    m_cs=zeros(J);
    y_cs=zeros(J);
    % update y^(n)
    for n=1:N
        y{n}=y{n}+mu*(m{n}-x);
    end
    % evaluate recovery accuracy
    RC(i)=norm(x(:)-x0(:),2)/norm(x0(:),2);
    RE(i)=norm(x(:)-tnsr(:),2)/norm(tnsr(:),2);
    if flag
        fprintf('Iteration=%d\tRC=%f\tRE=%f\n',i,RC(i),RE(i));
    end
    if RC(i)<epsilon_x
        break
    end
    x0=x;
end
run_time=cputime-t;
fprintf('running time=%fs\n',run_time);
end