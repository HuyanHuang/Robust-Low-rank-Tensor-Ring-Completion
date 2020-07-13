%% color video
clc
clear
close all
%% input arguments
SR=0.5;
NR=0.1;
set_method={'RTRC','TNN','SNN','RMC','SiLRTC-TT','STTC','TRNNM','FBCP'};
run_time=zeros(2,8);
x=cell(2,8);
% load run_time_video
% load recovery_video
%% solve
for i=1:2
    switch i
        case 1
            load visiontraffic
            vid=visiontraffic(:,:,:,101:156);
            siz=size(vid);
            vid=reshape(vid,[90,160,3,7,8]);
        case 2
            load bootstrap_trunc
            vid=bootstrap_trunc;
            siz=size(vid);
            vid=reshape(vid,[120,160,3,7,7]);
    end
    vid=double(vid);
    %% sampling
    P=sampling_uniform(vid,SR);
    %% adding sparse noise
    vid_noise=noise_sparse_P(vid,P,NR,2);
    %% solve problem
    % RTRC
    [x1,~,~,run_time(i,1)]=RTRC(vid_noise,P,10^-4,false);%4.4-4.6
    x{i,1}=reshape(x1,siz);
    % RTC-TNN
    t=cputime;
    T=reshape(vid_noise,[siz(1:2),prod(siz(3:end))]);
    x1=trpca_tnn(T,1/sqrt(size(T,3)*max(size(T,1),size(T,2))));
    run_time(i,2)=cputime-t;
    x{i,2}=reshape(x1,siz);
    % RTC-SNN
    t=cputime;
    x1=trpca_snn(vid_noise,[1,1,0.1,1,1]);
    run_time(i,3)=cputime-t;
    x{i,3}=reshape(x1,siz);
    % RMC
    t=cputime;
    mat=reshape(vid_noise,[],prod(siz(3:end)));
    x1=lrmcR(mat,find(P),1/sqrt(SR*max(size(mat))));
    run_time(i,4)=cputime-t;
    x{i,4}=reshape(x1,siz);
    % SiLRTC-TT
    t=cputime;
    [~,ranktube]=SVD_MPS_Rank_Estimation(P.*vid_noise,1e-2);
    x1=SiLRTC_TT(vid_noise,find(P),ranktube,0.01*ranktube,500,1e-5);
    run_time(i,5)=cputime-t;
    x{i,5}=reshape(x1,siz);
    % STTC
    V=reshape(vid_noise,siz);
    PP=reshape(P,siz);
    t=cputime;
    x{i,6}=Smoothlowrank_TV_video21(PP.*V,find(P),0,P.*vid_noise,[20,20,0,20]);
    run_time(i,6)=cputime-t;
    % TRNNM
    [x1,~,~,run_time(i,7)]=TRNNM(vid_noise,P,10^-5,false);
    x{i,7}=reshape(x1,siz);
    % FBCP
    t=cputime;
    model=BCPF_IC(P.*vid_noise,'obs',P,'init','rand','maxRank',10,'maxiters',100,...
        'tol',1e-4,'dimRed',1,'verbose',0);
    run_time(i,8)=cputime-t;
    x1=double(model.X);
    x{i,8}=reshape(x1,siz);
    %% visualize the results
    switch i
        case 1
            vid=reshape(vid,[90,160,3,56]);
            vid_noise=reshape(vid_noise,[90,160,3,56]);
            P=reshape(P,[90,160,3,56]);
            idx=49:56;
        case 2
            vid=reshape(vid,[120,160,3,49]);
            vid_noise=reshape(vid_noise,[120,160,3,49]);
            P=reshape(P,[120,160,3,49]);
            idx=1:8;
    end
    siz=size(vid(:,:,:,idx));
    x1=reshape(permute(vid(:,:,:,idx),[1,4,2,3]),[prod(siz([1,4])),siz(2),siz(3)]);
    x2=reshape(permute(vid_noise(:,:,:,idx),[1,4,2,3]),[prod(siz([1,4])),siz(2),siz(3)]);
    P_temp=reshape(permute(P(:,:,:,idx),[1,4,2,3]),[prod(siz([1,4])),siz(2),siz(3)]);
    x1=[x1,P_temp.*x2];
    x1=uint8(x1);
    figure;
    imshow(x1,'border','tight');
    if i==1
        saveas(gcf,'visiontraffic_observed.png');
    else
        saveas(gcf,'bootstrap_observed.png');
    end
    for j=1
        x1=reshape(permute(x{i,j}(:,:,:,idx),[1,4,2,3]),[siz(1)*length(idx),siz(2),siz(3)]);
        x1=uint8(x1);
        y=reshape(permute(vid(:,:,:,idx),[1,4,2,3]),[siz(1)*length(idx),siz(2),siz(3)]);
        y=uint8(y);
        y=imabsdiff(y,x1);
        x1=[x1,y];
        figure;
        imshow(x1,'border','tight');
        if i==1
            saveas(gcf,['visiontraffic_' set_method{j} '.png']);
        else
            saveas(gcf,['bootstrap_' set_method{j} '.png']);
        end
    end
end
save recovery_video x
save run_time_video run_time