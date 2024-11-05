function [beta, error, psnr_list, ssim_list] = ISTA(h, y, I, lambda, max_iter, tol, cases)
    % ISTA algorithm for LASSO problem
    % X: design matrix
    % y: observation vector
    % lambda: regularization parameter
    % max_iter: maximum number of iterations
    % tol: tolerance for convergence
beta=zeros(size(I));
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
error(1)=0.5*norm(H(beta)-y,'fro')^2;
psnr_list(1) = psnr(beta, I);
ssim_list(1) = ssim(beta, I); 

    for k = 1:max_iter
        % Gradient descent step
        grad = HT(H(beta) - y);
        if cases == 'L1'
            beta_new = prox_l1(beta - t * grad, lambda * t);
        elseif cases == 'TV'
            grads = beta - t * grad;
            beta_new = Condat_TV_1D_v2( grads(:), lambda * t);
            beta_new = reshape(beta_new, size(I));
        end
        
        % Convergence check
        if norm(beta_new - beta, 'fro') < tol
            break;
        end
        beta = beta_new;
        
        psnr_list(k+1) = psnr(beta, I);
        ssim_list(k+1) = ssim(beta, I); 
        error(k+1)=0.5*norm(H(beta)-y, 'fro')^2;
    end
end


