function v=sprandvec(m,n)

v = zeros(1,m);
r = randperm(m);
r = r(1:n);
for i=1:n
    v(r(i)) = 1;
end
v = v';
end
