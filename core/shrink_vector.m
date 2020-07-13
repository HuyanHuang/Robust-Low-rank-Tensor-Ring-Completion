function v1=shrink_vector(v0,tau)
v1=sign(v0).*max(abs(v0)-tau,0);
end