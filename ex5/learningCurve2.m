function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, iterations)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

m_val = length(yval);

for i = 1:m
    for j = 1:iterations
        lucky_train = randperm(m)(1:i);
        lucky_cv = randperm(length(y))(1:i);

        X_train = [ones(i,1), X(lucky_train,:)];
        y_train = y(lucky_train);
        X_cv = [ones(i,1), Xval(lucky_cv,:)];
        y_cv = yval(lucky_cv);

        [theta] = trainLinearReg(X_train, y_train, lambda);
        error_train(i) += linearRegCostFunction(X_train, y_train, theta, 0);
        error_val(i) += linearRegCostFunction(X_cv, y_cv, theta, 0);
    end

    error_train(i) /= iterations;
    error_val(i) /= iterations;
end

end
