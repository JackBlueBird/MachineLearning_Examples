% Author: Giacomo Zuccarino
% giacomozuccarino@gmail.com
%
% April 2023
%
% --- what is this about ---
% the aim of this script is to show how regularization can overcome
% problems associated with least square technique performed on matrix with
% singular eigenvalues.
%
% --- problem origin ---
% the context from which this problem origins is the one of multiple
% variables linear regression, which is performed solving a least square
% method over the data.
% multilinear regression is a key technique in modern Machine Learning
% 
% from a data matrix X and a target vector y we want to find the best
% linear approximation of y using X as a predictor
%   y = aX+b
% finding w = (a,b) is the equivalent of minimizing
% J = (1/N)*(Xw-y)'*(Xw-y)
% over all possible choices of w
%
% the problem can be solved in many ways:
% - in python, the package science kit learn provides a reliable solver for multiple linear regression
% - for not so large dataset, a gradient descent algorithm can be
% implemented almost in every computational framework
% - direct numerical computation, after some math, it is shown that the
% best possible choice of w is given by
% 
% w = (X'X)^(-1)*X'*y
%
% as consequence, the problem is reduced to finding the inverse of the
% matrix X'X. If X'X has singular eigenvalues both X'X and its inverse are
% ill conditioned and the numerical results of both computing the inverse
% and using it in the calculation can lead to wrong results very far away
% from those predicted from the theory of the problem.
%
% --- why is it of interest? ---
% of course, modern toolkits for ML deal with the problem efficiently and under the hood.
% Nevertheless, solving by hand a very small example should be within the
% skills of everyone approaching ML and AI. Yes, this problem cannot be
% solved directly without addressing the singular nature of its data.
%
% --- proposed approach ---
% our method, derived from a wide variety of examples coming from different
% topics in scientific computing,
% consists in diagonalizing the matrix X'X and using its diagonalization to
% perform the computation
% X'X = PDP' <=> X'X^(-1) = PD^(-1)P'
%
% since the diagonal is still singular, a regularization is performed
% D_reg = D +epsilon
% where epsilon is added only to the singular entries
% the corresponding solution
%
% w_reg = PD_reg^(-1)P'X'y
%
% ant is is compared with the ill conditioned solution w
% moreover, a comparison with results obtained with python science kit learn is presented 
% --- ---


% X : 3x4 matrix with predicting features
% y : 3x1 column vector with the target data
clear
clc
%
X = [2104, 5, 1, 45;...
    1416, 3, 2, 40;...
    852, 2, 1, 35];
y = [460; 232; 178];

N = size(X,1); % number of samples
M = size(X,2); % number of features

% column standardization
for j = 1:M 
     X(:,j) = standardize_fcn(X(:,j));
end

% adding an additional row of ones to X to take into account the intercept
% of the linear regression
X = [X,ones(N,1)]; 
XtX = X'*X; % XtX is the matrix with which we have to deal to solve the linear regression
condX = cond(X); 
condXtX = cond(XtX); % the conditioning number of XtX is very high
detXtX = det(XtX); % the determinant is of order 10^-30
display(['conditioning number for XtX is of order',' ','10^',num2str(log10(condXtX),'%.0f')])
display('---')

[P,D] = eig(XtX); % P matrix of eigenvectors, D diagonal matrix of eigenvalues
D1 = zeros(5,5); % set up of inverse for D
PtP = P'*P; % P is orthogonal, as consequence PtP is the identity matrix

for i = 1:M+1
    D1(i,i) = D(i,i)^-1;
end

w = P*D1*P'*X'*y; % solution vector computed with the inverse
Jw = 1/N*(X*w-y)'*(X*w-y); % minimum computed using w, quite high
Jw_b = 1/N*(w'*P*D*P'*w-2*w'*X'*y+y'*y); % another scheme for the minimum
display(['the minimum computed with the direct solution is',' ',num2str(Jw,'%.2f')])
display('---')


eps2 = 10^-5; %regularization parameter eps
XtXnew = XtX+P'*diag([eps2,eps2,0,0,0])*P; %diagonal regularization
condXtXnew = cond(XtXnew); % new conditioning number
detXtXnew = det(XtXnew); % new determinant

D_reg = D+diag([eps2,eps2,0,0,0]); % regularized diagonal
w_reg = P*inv(D_reg)*P'*X'*y; % regularized solution
Jw_reg = 1/N*(X*w_reg-y)'*(X*w_reg-y); % numerical scheme 1 for the minimum
Jw_regb = 1/N*(w_reg'*P*D*P'*w_reg-2*w_reg'*X'*y+y'*y); % numerical scheme 2 for the minimum
display(['the minimum computed with the regularized solution is',' ',num2str(Jw_reg,'%.2f')])
display('---')
% science kit learm solution
w_skl = [38.0516; 41.5433; -30.9889; 36.3418; 290 ]; % skl solution
Jw_skl = 1/N*(X*w_skl-y)'*(X*w_skl-y);
display(['the minimum computed with the python skl solution is',' ',num2str(Jw_skl,'%.2f')])
display('---')


diff_d_skl = abs(w_skl-w_reg);
display(['difference between current solution and python skl solution is of order',' ','10^',num2str(log10(mean(diff_d_skl)),'%.0f')])


function y = standardize_fcn(x)
N = length(x);
x = reshape(x,[N,1]);
mu = sum(x)/N;
sigma = sqrt((1/N)*(x-mu)'*(x-mu));
y = (x-mu)/sigma;
end