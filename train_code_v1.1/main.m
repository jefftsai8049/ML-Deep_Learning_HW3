%%Initialization
clear ; close all; clc
fprintf('\nLoading Files...\n')

% Setup the parameters 
type_of_word2vec  = 500;                           % 500 corresponds to -3 ~ 499, so input layer should have 503 nodes
vector_size       = type_of_word2vec + 3;
input_layer_size  = type_of_word2vec + 3;          % 503 nodes
hidden_layer_size = (type_of_word2vec + 3) * 2;    % 1006 hidden units
memory_size       = hidden_layer_size;             % 1006 hidden units
num_labels        = type_of_word2vec + 3;          % 503 labels



% =========== Part 1: Loading  Data =============

% Load Training Data
%%%%%%%%%%%%%%%[raw_y,raw_X] = input_train('train_out_map.txt');
number_of_sentences = 15;
data = importTrainData('training_index_word2vec500.csv',number_of_sentences);

%   ================ Part 2: Initializing Pameters ================
%  (randInitializeWeights.m)
f=[]
%for learning_rate=3:3
    %for t=7:7
        %hidden_layer_size= 128;   % 128 hidden units
        %hidden_layer_size = 128;   % 128 hidden units
        
        %parameters setting
        %batch = pow2(t);
        learning_rate = 3;
        yita = 0.01*learning_rate;   %learning rate
        decay = 0.9999;
        
        
        fprintf('\nInitializing Neural Network Parameters ...\n')

        [initial_W_1, initial_b_1] = randInitializeWeights(input_layer_size, hidden_layer_size);
        [initial_W_m, initial_b_m] = randInitializeWeights(memory_size, hidden_layer_size);
        [initial_W_out, initial_b_out] = randInitializeWeights(hidden_layer_size, num_labels);

        % Unroll parameters
        initial_nn_params = [initial_W_1(:) ; initial_b_1(:) ; initial_W_m(:) ; initial_b_m(:) ; initial_W_out(:) ; initial_b_out(:)];

        nn_params = initial_nn_params;

        % ================ Part 3: Compute Cost (Feedforward) ================
        
        fprintf('\nFeedforward Using Neural Network ...\n')

        % Weight regularization parameter
        lambda = 0;
        number_of_words = size(data{1}, 2);     % take first sentence as example
        whole_sentence = zeros(vector_size, number_of_words);
        
        for current_word = 1 : number_of_words
            index = data{1}(current_word) + 4;
            whole_sentence(index, current_word) = 1;
        end
        
        X = whole_sentence(:, 1:number_of_words-1);
        y = whole_sentence(:, 2:end);
        
        current_word = 1;       % current_word is a parameter to be used later for back prop.
        [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, memory_size , ...
                           num_labels, X, y, lambda, current_word);
                       
        while isnan(J)
            [initial_W_1, initial_b_1] = randInitializeWeights(input_layer_size, hidden_layer_size);
            [initial_W_m, initial_b_m] = randInitializeWeights(memory_size, hidden_layer_size);
            [initial_W_out, initial_b_out] = randInitializeWeights(hidden_layer_size, num_labels);

            % Unroll parameters
            initial_nn_params = [initial_W_1(:) ; initial_b_1(:) ; initial_W_m(:) ; initial_b_m(:) ; initial_W_out(:) ; initial_b_out(:)];

            nn_params = initial_nn_params;
            [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, memory_size , ...
                           num_labels, X, y, lambda, current_word);            
        end 

        fprintf(['Cost at initial parameters : %f '...
                 '\n\n'], J);
        

        fprintf('\nProgram paused. Press enter to continue.\n');
        % pause;

        % =================== Part 4: Training NN ===================
      
        fprintf('\nTraining Neural Network... \n')
       
        %  Weight regularization parameter
        lambda = 0;
        sigma = 0;              % Adagrad parameter
        current_iteration = 0;  % Adagrad parameter
        sum_of_gradients = zeros(size(nn_params, 1), 1);   % Adagrad parameter        

        for epoch = 1:1
                for current_sentence = 1 : number_of_sentences
                        
                        number_of_words = size(data{current_sentence}, 2);     % number of words the current sentence has 
                                                                                                 %(including start and period)
                        whole_sentence = zeros(vector_size, number_of_words);

                        for current_word = 1 : number_of_words
                            index = data{current_sentence}(current_word) + 4;
                            whole_sentence(index, current_word) = 1;
                        end

                        X = whole_sentence(:, 1:number_of_words-1);
                        y = whole_sentence(:, 2:end);
                                                                     
                        if number_of_words <= 3         % if number_of_words <= 3 , backprop. is impossible
                            continue;          
                        end
                        
                        for current_word = 1 : (number_of_words - 3)    % current_word is a parameter to indicate which y to back prop. from
                        
                             [J , grad] = nnCostFunction(nn_params, ...
                                                       input_layer_size, ...
                                                       hidden_layer_size, memory_size , ...
                                                       num_labels, ...
                                                       X, y, lambda, current_word);
                             
                             sum_of_gradients = sum_of_gradients + grad .^2;                      
                             sigma = sqrt( sum_of_gradients / (current_iteration + 1) );

                             nn_params =  nn_params   -   grad / sum(sum(sigma)) * yita;
                             current_iteration = current_iteration + 1;
                             
                        end
                        
                         fprintf(['Epoch : %d  Sentence : %d  '...
                                     ], epoch, current_sentence);
                         fprintf(['Sum of gradients : %f  '...
                                     ], sum(sum(grad)));
                         fprintf(['Cost : %f '...
                                     '\n\n'], J);
                                 
                end
                yita = decay * yita;
        end

        fprintf('Program paused. Press enter to continue.\n');
        % pause;

        % ================= Part 5: Implement Predict =================
        %  compute the training set accuracy

        %total_acc = 0;   %total accuracy

                %y = raw_y(:, (k*128+1):(k*128+3000));
                %X = raw_X(:, (k*128+1):(k*128+3000));

        input_samples = 10000;        

        %%%%%%%%%%%%%%%%%y = raw_y(:, 1  : input_samples);
        %%%%%%%%%%%%%%%%%X = raw_X(:,1 : input_samples);
        %{
        pred = predict(nn_params, input_layer_size, hidden_layer_size, memory_size, num_labels, X);

        pred_test = zeros(48,size(X,2));

        for i=1:size(X,2)
           pred_test(pred(1,i),i) = 1;
        end

        %total_acc =   mean(mean(double(pred_test == y))) * 100;
        %sum(sum(abs(A-B)));

        %total_acc = total_acc / (k + 1);

        fprintf('\nTraining Set Accuracy: %f\n', (100 - sum(sum(abs(pred_test  - y )))/ input_samples/2 * 100));
        msg = [yita,batch,(100 - sum(sum(abs(pred_test  - y )))/ input_samples/2 * 100)];
        f=[f;msg];
        %}
    %end
%end
% dlmwrite('accuracy.txt',f);

