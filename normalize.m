function [x] = normalize(x)

x_max = max(x);
x_min = min(x);

for i = 1:size(x)
    x(i) = (x(i) - x_min)/(x_max - x_min);
end

end
