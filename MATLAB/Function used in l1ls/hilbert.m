
function H = hilbert(m,n)
for i = 1:m 
    for j = 1:n  
        H(i, j) = 1/(i+j-1);
    end
end 
end