function P=sampling_uniform(tnsr,SR)
%% calculate the total number of tnsr
num=numel(tnsr);
%% uniformly random sampling
idx=randsample(num,ceil(SR*num));
order=size(tnsr);
P=zeros(order);
P(idx)=1;
end