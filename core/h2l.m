function tnsr=h2l(tnsr,I,order,siz)
tnsr=reshape(tnsr,I(order));
tnsr=ipermute(tnsr,order);
tnsr=reshape(tnsr,siz);
end