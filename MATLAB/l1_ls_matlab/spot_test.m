n = 1000; x = (1:n)';        % the first column defines a circulant matrix
F = opDFT(n);                % create a DFT operator
s = sqrt(n)*F*x;             % eigenvalues of C
C = real( F'*opDiag(s)*F );  % the circulant operator
w = C*x;                     % apply C to a vector
y = C'*w;                    % apply the adjoint of C to a vector
z = C(end:-1:1,:)*y;         % reverse the rows of C and apply to a vector
double( C(1:5,1) )'    