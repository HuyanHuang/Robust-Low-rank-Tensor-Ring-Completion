clc
clear
close all
%% input arguments
d=4;
n=20;
TRr=2:14;
SR=0.1:0.1:1;
NR=0:0.01:0.5;
err_reL=zeros(length(TRr),length(NR),length(SR));
num=1;
for k=1:length(SR)-1
    for j=1:length(NR)
        for i=1:length(TRr)
            tnsr=TR_rand(n*ones(1,d),TRr(i)*ones(1,d));
            for m=1:num
                %% sampling
                P=sampling_uniform(tnsr,SR(k));
                %% adding sparse noise
                tnsr_noise=noise_sparse_P(tnsr,P,NR(j));
                %% solve problem via ADMM
                if SR(k)~=1
                    x=RTRC(tnsr_noise,P,10^-2,false);
                else
                    x=rpca(reshape(tnsr_noise,n^(d/2),[]),1/n);
                end
                err_reL(i,j,k)=err_reL(i,j,k)+norm(x(:)-tnsr(:),2)/norm(tnsr(:),2);
            end
            err_reL(i,j,k)=err_reL(i,j,k)/num;
        end
    end
end
%% visualization
% idx=find(err_reL<=1e-3);
% vert=[0 0 0;1 0 0;1 1 0;0 1 0;0 0 1;1 0 1;1 1 1;0 1 1];
% vert(:,1)=vert(:,1);
% vert(:,2)=vert(:,2)*0.1;
% vert(:,3)=vert(:,3)*0.2;
% fac=[1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8;1 2 3 4;5 6 7 8];
% figure;
% hold on
% for m=1:length(idx)
%     [i,j,k]=ind2sub([length(TRr),length(NR),length(SR)],idx(m));
%     patch('Vertices',vert+repmat([TRr(i) NR(j) SR(k)],8,1),'Faces',fac,'FaceColor','b','FaceAlpha','0.8');
% end
% xlabel('TR-rank');
% ylabel('\gamma');
% zlabel('SR');
%% visualization
 err=ones(size(err_reL));
 err(:,:,end)=err_reL(:,:,end);
 for k=1:length(SR)-1
     thr=floor(sqrt(SR(k))*TRr)-1;
     for i=1:length(TRr)
         idx=find(thr==TRr(i));
         if ~isempty(idx)
             err(i,:,k)=mean(err_reL(idx,:,end),1);
         end
     end
 end
err=err_reL;
figure;
for i=1:length(SR)
    subplot(2,5,i);
    imagesc([TRr(1) TRr(end)],[NR(1) NR(end)],log10(err(:,:,i))',[-3 -2]);
    colormap(flipud(gray));
    set(gca,'YDir','normal');
    if i==6||i==7||i==8||i==9||i==10
        xlabel('TR-rank');
    end
    if i==1||i==6
        ylabel('\gamma');
    end
    title(['P=' num2str(SR(i))]);
end
% colorbar('Ticks',[-3 0],'TickLabels',{'<10^{-3}','10^{0}'},'Direction','reverse');