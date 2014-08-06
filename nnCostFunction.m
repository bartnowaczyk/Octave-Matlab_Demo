function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

    m = size(X, 1);
    J = 0;


    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

    	Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    y_matrix = eye(num_labels)(y,:);

    X = [ones(size(X), 1) X];
    z2 = Theta1*X';
    a2 = sigmoid(z2);
    a2 = a2';
    a2 = [ones(m, 1) a2];
    z3 = Theta2*a2';
    a3 = sigmoid(z3);
    h0 = a3';

    l1 = log(h0);
    l2 = log(1 - h0);
    s1 = -y_matrix .* l1;
    s2 =((1-y_matrix).*l2);
    s = s1 - s2;


    b = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2))+ sum(sum(Theta2(:, 2:end).^2))) ;
    z = 1/(m) * (sum(s));
    z =sum(z);
    J = z + b;


    d3 = y_matrix - h0;
    d2 = d3 * Theta2(:, 2:end);
    d2 = d2' .* sigmoidGradient(z2);

    D2 = d3' * a2;
    D1 = d2 * X;

    Theta1_grad = -(1/m) * D1;
    Theta2_grad = -(1/m) * D2;

    reg1 = (lambda/m)* Theta1;
    reg1(:,1) = 0;
    reg2 = (lambda/m)* Theta2;
    reg2(:,1) = 0;
    % disp(reg1);
    % disp(Theta1_grad);

    Theta1_grad = Theta1_grad + reg1;
    Theta2_grad = Theta2_grad + reg2;

    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
