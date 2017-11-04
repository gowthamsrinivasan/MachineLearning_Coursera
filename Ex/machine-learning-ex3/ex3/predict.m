function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m,1) X]; %X(5000x401)

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%%%% Sizes %%%
% X(5000x401)
% Theta1(25x401)
% Theta2(10x26)
%%%%%%%%%%%%%%

z1 = Theta1*X'; %z1(25x5000)
a1 = sigmoid(z1); %a1(25x5000) since z1(25x5000)
a1 = [ones(1,size(a1,2)) ;a1]; %(26x5000)
z2 = Theta2*a1; %z2(10x5000) since a1(26x5000)
a2 = sigmoid(z2); % a2(10x5000) since z2(10x5000)

[~, p] = max(a2,[],1); % p(1x5000) = Index of maximum of each column
p = p'; % p(5000x1); to match dimensions of y




% =========================================================================


end
