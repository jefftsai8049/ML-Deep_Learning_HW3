function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, memory_size ,  ...
                                   num_labels, ...
                                   X, y, lambda, current_word)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters, the weight matrices for RNN
current_index = 1;
current_size = hidden_layer_size * input_layer_size;   %  128 x 69
W_1 = reshape(nn_params(1 : current_size), ...
                 hidden_layer_size, input_layer_size);         

current_index = current_index + current_size;      % 1  + 128 x 69
current_size = hidden_layer_size ;                    %  128 x 1
b_1 = reshape(nn_params(  current_index   :     current_index + current_size - 1), ...
                 hidden_layer_size, 1 );           
 
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1
current_size = memory_size * hidden_layer_size  ;                    %  128 x 128
W_m = reshape(nn_params(current_index   :   current_index + current_size - 1  ), ...
                 memory_size, hidden_layer_size);           
 
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128
current_size = hidden_layer_size * 1  ;                    %  128 x 1
b_m = reshape(nn_params(current_index  :     current_index + current_size - 1  ), ...
                 hidden_layer_size, 1 );           
  
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128  +  128 x 128
current_size = num_labels * hidden_layer_size  ;                    %  48 x 128
W_out = reshape(nn_params( current_index  :  current_index + current_size - 1 ), ...
                 num_labels, hidden_layer_size);                   
   
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128  +  128 x 128  +  48 x 128
current_size = num_labels * 1  ;                    %  48 x 1
b_out = reshape(nn_params( current_index  :  end ), ...
                 num_labels, 1 );                   

% Setup some useful variables
number_of_words = size(X, 2) + 1;
         
% need to return the following variables
J = 0;
W_1_grad = zeros(size(W_1));
b_1_grad = zeros(size(b_1));
W_m_grad = zeros(size(W_m));
b_m_grad = zeros(size(b_m));
W_out_grad = zeros(size(W_out));
b_out_grad = zeros(size(b_out));


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

a_m = zeros(memory_size, 1);        % initial memory at the start should be 0
memory_all = zeros(memory_size, number_of_words - 1);    % save hidden layer values for back propagation
z_all = zeros(memory_size, number_of_words - 1);         % save hidden layer values for back propagation

for cur_word = 1 : (number_of_words - 1)    % current word
        
        y_cur_word = y( : , cur_word);        %  503 x 1      the ground truth for current word (should be a vector of zeros with one value = 1)
        X_cur_word = X( : , cur_word) ;       %  503 x 1
        z_1 = W_1 * X_cur_word + b_1 ...      % 
              +  W_m * a_m + b_m;         
        z_all(:, cur_word) = z_1; 
        a_1 = ReLU(z_1) ;                     % 128 x 1
        a_m = a_1;                            % copy value of hidden layer to memory
        memory_all(:, cur_word) = a_m;        % copy value of hidden layer to memory
        z_out = W_out * a_1 + b_out ;              % 48 x 1
        a_out = Softmax(z_out) ;                   % 48 x 1  (output)        
        
        if cur_word == (number_of_words - current_word)
            y_out = a_out;     % this is the y to be used for back propagation
        end
        
        t_hat =  (y_cur_word==1) ;          % find the index equal to 1
        yt = a_out(t_hat);         % corresponding value of a_out at the index
        if yt == 0
            cur_word_cost = -log(0.0001);  % cost for current word
        else
            cur_word_cost = -log(yt);  % cost for current word
        end
       
        J = J + cur_word_cost; % accumulate the cost
end

% Part 2: Back prop. according to the current step.

%============ Unfold a part of the RNN===================
starting_memory = number_of_words - current_word - 3;

if starting_memory == 0
    z_1_BPTT = zeros(memory_size, 1);
    layer_1 = zeros(memory_size, 1);    
else
    z_1_BPTT = z_all(:, starting_memory);
    layer_1 = memory_all(:, starting_memory);
end

z_2_BPTT = memory_all(:, starting_memory + 1);
z_3_BPTT = memory_all(:, starting_memory + 2);
z_4_BPTT = memory_all(:, starting_memory + 3);
layer_2 = memory_all(:, starting_memory + 1);
layer_3 = memory_all(:, starting_memory + 2);
layer_4 = memory_all(:, starting_memory + 3);
x_1 = X( : , starting_memory + 1);
x_2 = X( : , starting_memory + 2);
x_3 = X( : , starting_memory + 3);
y_hat_out = y(:, number_of_words - current_word);
% y_out is already saved during back prop.

%============ Unfolding complete ===================

delta_out = y_out;
t_hat = ( y_hat_out == 1 );                        % find the index equal to 1
delta_out(t_hat) = delta_out(t_hat) - 1;           % corresponding value of delta at t_hat should be y_out-1, 
                                                   %                    while the other values are just y_out.
delta_4 = ReLUGradient( z_4_BPTT) .*  ( W_out'  * delta_out);
delta_3 = ReLUGradient( z_3_BPTT) .*  ( W_m'  * delta_4);
delta_2 = ReLUGradient( z_2_BPTT) .*  ( W_m'  * delta_3);


W_out_grad = W_out_grad + delta_out * layer_4' ;    
b_out_grad = b_out_grad + delta_out ;               
W_m_grad = W_m_grad + delta_4 * layer_3' ;        % W_m_3
b_m_grad = b_m_grad + delta_4 ;                    
W_1_grad = W_1_grad + delta_4 * x_3' ;        % W_1_3
b_1_grad = b_1_grad + delta_4 ;       
W_m_grad = W_m_grad + delta_3 * layer_2' ;        % W_m_2
b_m_grad = b_m_grad + delta_3 ;                    
W_1_grad = W_1_grad + delta_3 * x_2' ;        % W_1_2
b_1_grad = b_1_grad + delta_3 ;    
W_m_grad = W_m_grad + delta_2 * layer_1' ;        % W_m_1
b_m_grad = b_m_grad + delta_2 ;                    
W_1_grad = W_1_grad + delta_2 * x_1' ;        % W_1_1
b_1_grad = b_1_grad + delta_2 ;    
        

%+  lambda/2/m * ( sum(sum(W_1( : , : ) .^2 )) + sum(sum(W_out( : , : ) .^2 ))  + sum(sum(b_1( : , : ) .^2 ))  + sum(sum(b_out( : , : ) .^2 ))  );
%{
%}

%  Implement regularization with the cost function and gradients.
% Compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%{
regterm_W_1 = lambda/number_of_words * W_1;
regterm_W_out = lambda/number_of_words * W_out;
regterm_b_1 = lambda/number_of_words * b_1;
regterm_b_out = lambda/number_of_words * b_out;
}%
%W_1_grad = W_1_grad + regterm_W_1;
%W_out_grad = W_out_grad + regterm_W_out;
%b_1_grad = b_1_grad + regterm_b_1;
%b_out_grad = b_out_grad + regterm_b_out;

% -------------------------------------------------------------

% =========================================================================
%}
% Unroll gradients
grad = [W_1_grad(:) ; b_1_grad(:) ;W_m_grad(:) ; b_m_grad(:) ; W_out_grad(:) ; b_out_grad(:)];


end
