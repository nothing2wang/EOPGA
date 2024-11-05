function [x_new, error, psnr_list, ssim_list] = EOptISTA(h, y, I, lambda, max_iter, tol, cases)
% OGM algorithm for LASSO problem
% X: design matrix
% y: observation vector
% lambda: regularization parameter
% max_iter: maximum number of iterations
% tol: tolerance for convergence



x_old = zeros(size(I));  % initialize beta
y_old = x_old;
z_old = x_old;

[n1,n2,n3] = size(y); % n1--rows, n2--cols, n3--layers

siz = size(h);  center = [fix(siz(1)/2+1),fix(siz(2)/2+1)];
P   = zeros(n1,n2,n3); 
for i=1:n3;  P(1:siz(1),1:siz(2),i) = h;  end
D  = fft2(circshift(P,1-center)); 
H  = @(x) (ifft2(D.*fft2(x)));         %%%% Blur operator.  B x 
HT = @(x) (ifft2(conj(D).*fft2(x)));   %%%% Transpose of blur operator.


D2=abs(D).^2;
L = norm(D2);  % Lipschitz constant (largest eigenvalue of X'X)
t = 1 / L;  % step size

Wn = 0;
pp=12;
for n=1:pp
    Wn = Wn+ nchoosek(pp,n)*(-t*D2)^(n-1);
end
theta(1)=1;
for k=2:max_iter
        theta(k) = (1+sqrt(1+4*theta(k-1)^2))/2;
end
theta(max_iter+1)= (1+sqrt(1+8*theta(k-1)^2))/2;
error(1)=0.5*norm(H(x_old)-y,'fro')^2;
psnr_list(1) = psnr(x_old, I);
ssim_list(1) = ssim(x_old, I); 
% Initialize variables
for k = 1:max_iter
    % Gradient descent step
    grad = HT(H(x_old) - y);
    gamma(k)=2*theta(k)/theta(max_iter+1)^2*(theta(max_iter+1)^2-2*theta(k)^2+theta(k));
        

    % Soft-thresholding operator
    if cases == 'L1'
        y_new = prox_l1(y_old - gamma(k)*t *Wn* grad, lambda * t*pp);
    elseif cases == 'TV'
            grads = y_old - gamma(k)*t *Wn* grad;
            y_new = Condat_TV_1D_v2( grads(:), lambda * t);
            y_new = reshape(y_new, size(I));
        end
    % Compute momentum parameter
    z_new = x_old + 1/gamma(k)*(y_new- y_old);
    
    % Optimized gradient update step
    x_new = z_new + ((theta(k) - 1) / theta(k+1)) * (z_new - z_old)+theta(k)/theta(k+1)*(z_new-x_old);

    % Convergence check
    if norm(y_new - y_old, 'fro') < tol
        break;
    end

    % Update variables for next iteration
    x_old = x_new;
    y_old = y_new;
    z_old = z_new;

    psnr_list(k+1) = psnr(x_old, I);
    ssim_list(k+1) = ssim(x_old, I); 
    error(k+1)=0.5*norm(H(x_old)-y,'fro')^2;
end
end