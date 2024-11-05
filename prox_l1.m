function x = prox_l1(x, lambda)
    % Soft-thresholding operator
    x = sign(x) .* max(abs(x) - lambda, 0);
end