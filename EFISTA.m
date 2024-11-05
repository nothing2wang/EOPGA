function [beta, error, psnr_list, ssim_list] = EFISTA(h, y, I, lambda, max_iter, tol, cases)
% OGM algorithm for LASSO problem
% X: design matrix
% y: observation vector
% lambda: regularization parameter
% max_iter: maximum number of iterations
% tol: tolerance for convergence


beta = zeros(size(I)); %zeros(p, 1);  % initialize beta
beta_old = beta;
z = beta;
theta_old = 1;

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
error(1)=0.5*norm(H(beta)-y,'fro')^2;
psnr_list(1) = psnr(beta, I);
ssim_list(1) = ssim(beta, I); 
% Initialize variables
for k = 1:max_iter
    % Gradient descent step
    grad = HT(H(z) - y);
    % Soft-thresholding operator
    if cases == 'L1'
        beta_new = prox_l1(z - t *Wn* grad, lambda * t*pp);
    elseif cases == 'TV'
            grads = z - t *Wn* grad;
            beta_new = Condat_TV_1D_v2( grads(:), lambda * t);
            beta_new = reshape(beta_new, size(I));
        end
    % Compute momentum parameter
    theta_new = (1 + sqrt(1 + 4 * theta_old^2)) / 2;

    % Optimized gradient update step
    %a = k/(k+3); b = (k+2)/(k+3);
    %z = beta_new + a * (beta_new - beta_old);% + b*(beta_new-z);
    z = beta_new + ((theta_old - 1) / theta_new) * (beta_new - beta_old);%+ theta_old/theta_new*(beta_new-z);

    % Convergence check
    if norm(beta_new - beta, 'fro') < tol
        break;
    end

    % Update variables for next iteration
    beta_old = beta_new;
    beta = beta_new;
    theta_old = theta_new;

    psnr_list(k+1) = psnr(beta, I);
    ssim_list(k+1) = ssim(beta, I); 
    error(k+1)=0.5*norm(H(beta)-y, 'fro')^2;
end
end