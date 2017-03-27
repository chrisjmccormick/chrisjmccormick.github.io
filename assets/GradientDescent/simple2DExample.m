% Record theta values over 10 iterations.
x = zeros(10, 1);

% Set the first theta value to 3.
x(1) = 3;

% Run ten iterations of gradient descent.
alpha = 0.1;
for (i = 2 : 10)
  x(i) = x(i - 1) - (alpha * 2 * x(i - 1));
end

% Plot the cost function.
theta = [-2:0.1:4]';
y = theta .^2;

plot(theta, y)
hold on;

% Plot the gradient descent thetas.
plot(x, x.^2, 'kx')

title('Gradient Descent')
xlabel('theta')
ylabel('J(theta)')
