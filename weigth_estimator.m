function [Neural_Network, m] = weigth_estimator(x, y, H)

[dim, nodes_in] = size(x);                            %I'm passing the argoument as a matrix, 
                                                      %the number of inputs correspond to the number of columns
W = unifrnd(0, 0.5, nodes_in+1, H);
K = unifrnd(0, 0.5, H, 1);

y_out = normalize(y);                                 %Normalization of the inputs and of the outputs 
x_in = normalize(x);                                  %to avoid problems in the sigmoid function
x_in = [ones(dim, 1) x_in];
                                                                      
sigmoid = @(v) 1./(1+exp(-v));                        %Sigmoid Function


tol = 1e-6;                                           %Setting the initial Tollerance to a small value
SSE_old = 10;                                         %Setting the SSE to an initial high value

bias = unifrnd(0, 0.5);                               %Setting the bias node to an initial value

max_iter = 1e5;                                       %creating a max number of iterations
mu_0 = 0.05;
mu = mu_0;

disp('###  STARTING THE WEIGTH ESTIMATOR  ###')

v = zeros(dim, H);

for m = 1:max_iter
     
    out = zeros(nodes_in, 1);                         %Output Nodes

    for p = 1:dim
    
        v_star = x_in(p, :)*W;                        %I'm creating now the hidden nodes value
        v(p, :) = sigmoid(v_star);                    %The second step is creating a non linear relation between in and out
                                                      %through the sigmoid function

        out(p) = v(p, :)*K + bias;                    %Output of my neural network
        
    end
    out = out';  
    SSE_new = (norm(out - y_out, 2))^2;               %Updating the SSE value to use it later as a stopping criteria
    err = abs(SSE_old - SSE_new);
    res = (y_out - out);                              %Residual between the input and the output
  

    [K, W, bias] = update_weigths(K, W, ...           %Updating the Weigths of all the network 
                x_in, mu, v, res, bias);
    mu = mu_0*0.99^(sqrt(m));

    if err <= tol
        break
    end
    SSE_old = SSE_new;
    if isnan(res)                                     %Check to avoid having a NaN vector for the residual
        break
    end

end

disp('###   NEURAL NETWORK COMPLETED   ###')
Neural_Network  = struct('weigths_in', W, 'weigths_out', K, 'act_func', sigmoid, 'bias', bias);

end



