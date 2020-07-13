function tnsr=noise_sparse_F(tnsr,NR,flag)
switch flag
    case 2
        %% non-uniform sparse noise (mode-2)
        siz=size(tnsr);
        tnsr=reshape(tnsr,[],prod(siz(3:end)));
        P=sampling_uniform(tnsr(:,1),NR);
        pixel_rand=randi(256,sum(P==1),1)-1;
        for i=1:prod(siz(3:end))
            tnsr(P==1,i)=pixel_rand;
        end
        tnsr=reshape(tnsr,siz);
    case 3
        %% uniform sparse noise (mode-3)
        P=sampling_uniform(tnsr,NR);
        pixel_rand=randi(256,sum(P(:)==1),1)-1;    
        tnsr(P==1)=pixel_rand;
end
end