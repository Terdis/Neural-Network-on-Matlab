function [K, W, bias] = update_weigths(K, W, X, mu, v, res, bias)

[N, H] = size(W);
n = size(res);

for i = 1:N
    for h = 1:H
        tmp = 0;
        for p = 1:n 
            tmp = tmp + res(p)*K(h)*v(p, h)*(1-v(p, h))*X(p, i);
        end
        W(i, h) = W(i, h) + 2*mu*tmp;
    end
end

for h = 1:H
    tmp = 0;
    for p = 1:n 
        tmp = tmp + res(p)*v(p, h);
    end
    K(h) = K(h) + 2*mu*tmp;
end

bias = bias + mu*sum(res);
end