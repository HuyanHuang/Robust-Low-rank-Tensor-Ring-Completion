function tnsr=l2h(tnsr,I,order,J)
tnsr=reshape(tnsr,I);
tnsr=permute(tnsr,order);
tnsr=reshape(tnsr,J);
end