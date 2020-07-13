clc
clear
close all
%% input arguments
d=4;
n=20;
TRr=5;
SR=0.81;
NR=0.1;
tnsr=TR_rand(n*ones(1,d),TRr*ones(1,d));
%% sampling
P=sampling_uniform(tnsr,SR);
%% adding sparse noise
tnsr_noise=noise_sparse_P(tnsr,P,NR,1);
E=tnsr_noise-tnsr;
%% solve problem via ADMM
[x,y,RC,run_time]=RTRC(tnsr_noise,P,10^-2,false);
%% evaluation
err_reL=norm(x(:)-tnsr(:),2)/norm(tnsr(:),2)
err_reS=norm(y(:)-E(:),2)/norm(E(:),2)
nnz(y)
nnz(E)
% %% visualize the results
% figure(1);
% semilogy(RC);
% legend('low-rank part','sparse part');
% xlabel('Iteration');
% ylabel('Relative Change');