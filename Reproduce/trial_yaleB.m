%% YaleB face
clc
clear
close all
%% input arguments
SR=0.5;
set_img={'yaleB1.mat','yaleB2.mat'};
set_method={'RTRC','TNN','SNN','RMC','SiLRTC-TT','STTC','TRNNM','FBCP'};
run_time=zeros(2,8);
x=cell(2,8);
% load run_time_YaleB
% load recovery_YaleB
%% solve
for i=1:length(set_img)
    img=importdata(set_img{i});
    img=img(:,:,[1:12 30:34 36:50]);
    siz=size(img);
    img=double(img);
    %% sampling
    P=sampling_uniform(img,SR);
    %% solve problem
    % RTRC
    [x{i,1},~,~,run_time(i,1)]=RTRC(img,P,10^-4,false);
    % RTC-TNN
    t=cputime;
    x{i,2}=trpca_tnn(P.*img,1/sqrt(SR*siz(3)*max(siz(1:2))));
    run_time(i,2)=cputime-t;
    % RTC-SNN
    t=cputime;
    x{i,3}=trpca_snn(P.*img,[1,1,1]);
    run_time(i,3)=cputime-t;
    % RMC
    t=cputime;
    mat=reshape(P.*img,[],siz(3));
    x1=lrmcR(mat,find(P),1/sqrt(SR*max(size(mat))));
    run_time(i,4)=cputime-t;
    x{i,4}=reshape(x1,siz);
    % SiLRTC-TT
    t=cputime;
    [~,ranktube]=SVD_MPS_Rank_Estimation(P.*img,1e-2);
    x{i,5}=SiLRTC_TT(img,find(P),ranktube,0.01*ranktube,500,1e-5);
    run_time(i,5)=cputime-t;
    % STTC
    t=cputime;
    x{i,6}=Smoothlowrank_TV12(P.*img,find(P),0,P.*img,[10,10,0]);
    run_time(i,6)=cputime-t;
    % TRNNM
    [x{i,7},~,~,run_time(i,7)]=TRNNM(img,P,10^-5,false);
    % FBCP
    t=cputime;
    model=BCPF_IC(P.*img,'obs',P,'init','rand','maxRank',30,'maxiters',100,...
        'tol',1e-4,'dimRed',1,'verbose',0);
    run_time(i,8)=cputime-t;
    x{i,8}=double(model.X);
    %% visualize the results
    idx=[10:12,27:32];
    x1=reshape(permute(img(:,:,idx),[1,3,2]),[],siz(2));
    P_temp=reshape(permute(P(:,:,idx),[1,3,2]),[],siz(2));
    x1=[x1,P_temp.*x1];
    x1=uint8(x1);
    figure;
    imshow(x1,'border','tight');
    if i==1
        saveas(gcf,'YaleB1_observed.png');
    else
        saveas(gcf,'YaleB2_observed.png');
    end
    for j=1
        x1=reshape(permute(x{i,j}(:,:,idx),[1,3,2]),[],siz(2));
        x1=uint8(x1);
        y=reshape(permute(img(:,:,idx),[1,3,2]),[],siz(2));
        y=uint8(y);
        y=imabsdiff(y,x1);
        x1=[x1,y];
        figure;
        imshow(x1,'border','tight');
        if i==1
            saveas(gcf,['YaleB1_' set_method{j} '.png']);
        else
            saveas(gcf,['YaleB2_' set_method{j} '.png']);
        end
    end
end
save recovery_YaleB x
save run_time_YaleB run_time