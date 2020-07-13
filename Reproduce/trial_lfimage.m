%% light-field image
clc
clear
close all
%% input arguments
SR=0.3;
NR=0.1;
set_lfimg={'greek.mat','medieval2.mat','pillows.mat','vinyl.mat'};
% I=[2*ones(1,14) 3*ones(1,5)];
% order=[1 8 2 9 3 10 4 11 5 12 6 13 7 14 15 16 17 18 19];
% J=[4 4 4 2 2 4 4 4 3*ones(1,5)];
% err_psnr=zeros(4,8);
% err_time=zeros(4,8);
load psnr_lfimage
load run_time_lfimage
%% solve
for i=1:1%length(set_lfimg)
    lfimg=importdata(set_lfimg{i});
    siz=size(lfimg);
    lfimg=double(lfimg);
    %% sampling
    P=sampling_uniform(lfimg,SR);
    %% adding sparse noise
    img_noise=noise_sparse_P(lfimg,P,NR,2);
    %% solve problem
    % RTRC
    [x,~,~,run_time]=RTRC(img_noise,P,10^-4,false);
    err_time(i,1)=run_time;
    err_psnr(i,1)=psnr(uint8(x),uint8(lfimg));
     % TNN
     t=cputime;
     x=lrtc_tnn(reshape(img_noise,[siz(1),siz(2),prod(siz(3:end))]),find(P==1));
     err_time(i,2)=cputime-t;
     x=reshape(x,siz);
     err_psnr(i,2)=psnr(uint8(x),uint8(lfimg));
     % SNN
     t=cputime;
     x=lrtc_snn(img_noise,find(P==1),[1,1,0.1,1]);
     err_time(i,3)=cputime-t;
     err_psnr(i,3)=psnr(uint8(x),uint8(lfimg));
     % RMC
     t=cputime;
     x=zeros(siz);
     for j=1:siz(end-1)
         for k=1:siz(end)
             x(:,:,j,k)=lrmc(img_noise(:,:,j,k),find(P(:,:,j,k)==1));
         end
     end
     err_time(i,4)=cputime-t;
     err_psnr(i,4)=psnr(uint8(x),uint8(lfimg));
     % SiLRTC_TT
     t=cputime;
     [~,ranktube]=SVD_MPS_Rank_Estimation(P.*img_noise,1e-2);
     x=SiLRTC_TT(img_noise,find(P),ranktube,0.01*ranktube,500,1e-5);
     err_time(i,5)=cputime-t;
     err_psnr(i,5)=psnr(uint8(x),uint8(lfimg));
     % STTC
     t=cputime;
     x=Smoothlowrank_TV_video21(P.*img_noise,find(P),0,P.*img_noise,[20,20,0,20]);
     err_time(i,6)=cputime-t;
     err_psnr(i,6)=psnr(uint8(x),uint8(lfimg));
     % TRNNM
     [x,~,~,run_time]=TRNNM(img_noise,P,10^-5,false);
     err_time(i,7)=run_time;
     err_psnr(i,7)=psnr(uint8(x),uint8(lfimg));
     % FBCP
     t=cputime;
     model=BCPF_IC(P.*img_noise,'obs',P,'init','rand','maxRank',30,'maxiters',100,...
         'tol',1e-4,'dimRed',1,'verbose',0);
     err_time(i,8)=cputime-t;
     x=double(model.X);
     err_psnr(i,8)=psnr(uint8(x),uint8(lfimg));
end
%%
save psnr_lfimage err_psnr
save run_time_lfimage err_time
%% visualize the results
color=[0 0 1;0 1 0;1 0 0;1 0 1;0 1 1;1 1 0;0.75 0.75 0.75;0 0 0];
figure(1);
subplot(2,1,1);
c=bar(err_psnr);
for i=1:8
    set(c(i),'FaceColor',color(i,:));
end
xlabel('Image index');
ylabel('PSNR (dB)');
legend('RTRC','RTC-TNN','RTC-SNN','RMC','SiLRTC-TT','STTC','TRNNM','FBCP','Orientation','horizontal');
legend('boxoff');
subplot(2,1,2);
c=bar(err_time);
for i=1:8
    set(c(i),'FaceColor',color(i,:));
end
xlabel('Image index');
ylabel('CPU time (s)');