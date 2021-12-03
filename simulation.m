clear all;
clc;

%%
%In this section we try to simulate the machine maintenance problem

n_sim = 1e4;
n = 20;
sim = ones(n_sim, 1);

N = 1000*sim;
W0 = 990*sim;
pB0 = 0.4;
pB1 = 0.001;
pR = 0.5;
C = 4;

I0 = 1;
I1 = 0.6;
pB = @(t) pB1*t + pB0*(1-t);
I = @(t) I1*t + I0*(1-t);
sp = linspace(0, 1, n)';
R = binornd(N-W0, pR);
income = zeros(n,1);

for i = 1:n
    m = sp(i);
    B = binornd(W0, pB(m));
    M = (W0 - B + R)*I(m) - C*R;
    income(i) = sum(M)/n_sim;
end

%%
%In this we must extract some of the data to train our Neural Network. 60
%percent of our data should be a sufficient training.

train_dim = int32(0.7*n);

index = randperm(n, train_dim);
x_train = zeros(train_dim, 1);
y_train = zeros(train_dim, 1);
index_c = zeros(n-train_dim, 1);

for i = 1:train_dim
    x_train(i) = sp(index(i));
    y_train(i) = income(index(i));
end

t = 1;
for i = 1:n
    flag = 0;
    for j = 1:train_dim
        if i == index(j)
            flag = 1;
        end
    end
    if(flag == 0)
        index_c(t) = i;
        t = t+1;
    end
end
        
x_test = zeros(n-train_dim, 1);
y_test = zeros(n-train_dim, 1);

for i = 1:(n-train_dim)
    x_test(i) = sp(index_c(i));
    y_test(i) = income(index_c(i));
end

%%
%In this Section we train our neural network and inizialise the number of
%hidden nodes. I am assuming that we want only 1 hidden layer

H = 4;
[NN, m] = weigth_estimator(x_train, y_train, H);

%%

x_test = normalize(x_test);
y_test = normalize(y_test);

x_test = [ones(n-train_dim, 1) x_test];

v = NN.act_func(x_test*NN.weigths_in);
out = v*NN.weigths_out + NN.bias;

fig1 = figure();
plot(x_test, y_test, 'LineWidth', 2, 'Marker', 'o');
hold on
plot(x_test, out, 'LineWidth', 2, 'Marker', 'x');
legend('Original', 'NN')
title('Average Income Depending on the Maintenance Level m')
xlabel('Maintenance Level')
ylabel('Average Income')
fig1.Children(1).FontSize = 20;
fig1.Children(2).FontSize = 20;
hold off

disp(['ITERATIONS OCCURRED: ' m])


