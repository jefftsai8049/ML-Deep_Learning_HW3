function p = predict(nn_params, ...
                     input_layer_size, ...
                     hidden_layer_size, memory_size ,  ...
                     num_labels,X)

input_layer_size = 69;
hidden_layer_size_1 = 128;
hidden_layer_size_2 = 128;

num_labels = 48;

current_index = 1;
current_size = hidden_layer_size_1 * input_layer_size;   %  128 x 69
W_1 = reshape(nn_params(1 : current_size), ...
                 hidden_layer_size_1, input_layer_size);         

current_index = current_index + current_size;      % 1  + 128 x 69
current_size = hidden_layer_size_1 ;                    %  128 x 1
b_1 = reshape(nn_params(  current_index   :     current_index + current_size - 1), ...
                 hidden_layer_size_1, 1 );           
 
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1
current_size = hidden_layer_size_2 * hidden_layer_size_1  ;                    %  128 x 128
W_2 = reshape(nn_params(current_index   :   current_index + current_size - 1  ), ...
                 hidden_layer_size_2, hidden_layer_size_1);           
 
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128
current_size = hidden_layer_size_2 * 1  ;                    %  128 x 1
b_2 = reshape(nn_params(current_index  :     current_index + current_size - 1  ), ...
                 hidden_layer_size_2, 1 );           
  
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128  +  128 x 128
current_size = num_labels * hidden_layer_size_2  ;                    %  48 x 128
W_out = reshape(nn_params( current_index  :  current_index + current_size - 1 ), ...
                 num_labels, hidden_layer_size_2);                   
   
current_index = current_index + current_size;      % 1  + 128 x 69  +  128 x 1  +  128 x 128  +  128 x 128  +  48 x 128
current_size = num_labels * 1  ;                    %  48 x 1
b_out = reshape(nn_params( current_index  :  end ), ...
                 num_labels, 1 );         
% Useful values

% You need to return the following variables correctly 
p = zeros( 1, size(X, 2));

%h1 = sigmoid([ones(m, 1) X] * Theta1');
%h2 = sigmoid([ones(m, 1) h1] * Theta2');

        z_1 = W_1 * X + repmat(b_1 , 1 ,   size(X,2)  );
        %z_1 = X' * W_1'  + repmat(b_1',   size(X',1) , 1   );   
        a_1 = sigmoid(z_1) ;     
        
        z_2 = W_2 * a_1 + repmat(b_2 , 1 ,   size(a_1,2)  );
        %z_1 = X' * W_1'  + repmat(b_1',   size(X',1) , 1   );   
        a_2 = sigmoid(z_2) ; 
        
        z_out = W_out * a_2 + repmat(b_out,    1  , size(a_2,2) );
        %z_2 = a_1' * W_out'  + repmat(b_out',   size(a_1',1) , 1   );                     
        a_out = sigmoid(z_out) ;                     
        
        
%[dummy, p] = max(a_2, [], 1);
[dummy, p] = max(a_out);

% =========================================================================


end
