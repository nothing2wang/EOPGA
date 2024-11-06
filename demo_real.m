close all;
clear;
clc;
fprintf('This is an example of Deblurring the blurred image \n');

I = double(imread('baboon512.png'))/255;



%h = fspecial('disk',5); 
h = fspecial('gaussian',10,20);
x0 = imfilter(I,h,'circular');
sigma=1e-4;
xx0 = x0 + sigma*randn(size(x0));
figure, imshow(xx0);
%%

lambda = 0.0001; % 0.0001 for L1 norm and TV norm
max_iter=300;
tol = 10^(-4);
%cases = 'L1';
cases = 'TV';
[n1, n2, n3]=size(I);

for ii=1:n3

%%
[beta_01(:,:,ii),error_01(ii,:), psnr_list_01(ii,:), ssim_list_01(ii,:)] = ISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
A(ii,:)=[min(error_01(ii,:)), max(psnr_list_01(ii,:)), max(ssim_list_01(ii,:))];

%%
[beta_02(:,:,ii), error_02(ii,:), psnr_list_02(ii,:), ssim_list_02(ii,:)] = FISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
B(ii,:)=[min(error_02(ii,:)), max(psnr_list_02(ii,:)), max(ssim_list_02(ii,:))];

%%
[beta_03(:,:,ii), error_03(ii,:), psnr_list_03(ii,:), ssim_list_03(ii,:)] = EFISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
C(ii,:)=[min(error_03(ii,:)), max(psnr_list_03(ii,:)), max(ssim_list_03(ii,:))];

%%
[beta_04(:,:,ii), error_04(ii,:), psnr_list_04(ii,:), ssim_list_04(ii,:)] = EOPGM(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
D(ii,:)=[min(error_04(ii,:)), max(psnr_list_04(ii,:)), max(ssim_list_04(ii,:))];

%%
[beta_05(:,:,ii), error_05(ii,:), psnr_list_05(ii,:), ssim_list_05(ii,:)] = EOptISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
E(ii,:)=[min(error_05(ii,:)), max(psnr_list_05(ii,:)), max(ssim_list_05(ii,:))];

end

results_all=[mean(A,1); mean(B,1); mean(C,1); mean(D,1); mean(E,1)]



figure;
semilogy(mean(error_01,1),'go-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(error_02,1),'b+-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(error_03,1),'m*-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(error_04,1),'kx-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); hold on
semilogy(mean(error_05,1),'rd-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); 
xlabel('Iterations');
ylabel('Tol');
legend('ISTA', 'FISTA', 'EFISTA', 'EOPGM' ,'EOptISTA');
set(gca,'Fontsize',20)


figure;
semilogy(mean(psnr_list_01,1),'go-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(psnr_list_02,1),'b+-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(psnr_list_03,1),'m*-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(psnr_list_04,1),'kx-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); hold on
semilogy(mean(psnr_list_05,1),'rd-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); 
xlabel('Iterations');
ylabel('PSNR');
legend('ISTA', 'FISTA', 'EFISTA', 'EOPGM' ,'EOptISTA');
set(gca,'Fontsize',20)
