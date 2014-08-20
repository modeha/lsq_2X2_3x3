function [w,Q] =gaussian(n,m)
if n<=m
    error('n should be greater than m')
end
w = wgn(m, 1, 0, 'dBm');
Q = rand(n,m);
end