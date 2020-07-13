function tnsr=noise_sparse_P(tnsr,P,NR,mode)
switch mode
    case 1
        %% uniformly sparse noise for synthetic data
        idx1=find(P==1);
        idx2=randsample(idx1,round(NR*length(idx1)));
        W=zeros(size(tnsr));
        W(idx2)=1;
        E=ones(size(tnsr));
        idx3=logical(sampling_uniform(tnsr,0.5));
        E(idx3)=-1;
        E=W.*E;
        tnsr=tnsr+E;
    case 2
        %% uniformly sparse noise for real-world data
        idx1=find(P==1);
        idx2=randsample(idx1,round(NR*length(idx1)));
        tnsr(idx2)=randi(256,1,length(idx2))-1;
end