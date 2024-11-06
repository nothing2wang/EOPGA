close all;
clear;
clc;
%clc;
fprintf('This is an example of Deblurring the blurred image \n');

%I = double(imread('baboon512.png'))/255;
%I = double(imread('peppers512.png'))/255;
%I = double(imread('fingerprint512.png'))/255;
%I = double(imread('barbarargb.png'))/255;
%I = double(imread('brickrgb.png'))/255;
I = double(imread('test_gd.png'))/255;


%h = fspecial('disk',5); 
h = fspecial('gaussian',10,20);
x0 = imfilter(I,h,'circular');
sigma=1e-4;
xx0 = x0 + sigma*randn(size(x0));
figure, imshow(xx0);
%%

lambda = 0.0001; % 0.0001 for L1 norm and TV norm
max_iter=300;
tol = 10^(-6);
%cases = 'L1';
cases = 'TV';
[n1, n2, n3]=size(I);
%I = I(1:min(n1,n2), 1:min(n1,n2),:);

imwrite(I, 'barbarargb.png')
for ii=1:n3

%%
[beta_01(:,:,ii),error_01(ii,:), psnr_list_01(ii,:), ssim_list_01(ii,:)] = ISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
%figure, imshow(beta_01)
A(ii,:)=[min(error_01(ii,:)), max(psnr_list_01(ii,:)), max(ssim_list_01(ii,:))];

%%
[beta_02(:,:,ii), error_02(ii,:), psnr_list_02(ii,:), ssim_list_02(ii,:)] = FISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
%figure, imshow(beta_02)
B(ii,:)=[min(error_02(ii,:)), max(psnr_list_02(ii,:)), max(ssim_list_02(ii,:))];

%%
[beta_03(:,:,ii), error_03(ii,:), psnr_list_03(ii,:), ssim_list_03(ii,:)] = EFISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
%figure, imshow(beta_03)
C(ii,:)=[min(error_03(ii,:)), max(psnr_list_03(ii,:)), max(ssim_list_03(ii,:))];

%%
[beta_04(:,:,ii), error_04(ii,:), psnr_list_04(ii,:), ssim_list_04(ii,:)] = EOPGM(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
%figure, imshow(beta_04)
D(ii,:)=[min(error_04(ii,:)), max(psnr_list_04(ii,:)), max(ssim_list_04(ii,:))];

%%
[beta_05(:,:,ii), error_05(ii,:), psnr_list_05(ii,:), ssim_list_05(ii,:)] = EOptISTA(h, xx0(:,:,ii), I(:,:,ii), lambda, max_iter, tol, cases);
%figure, imshow(beta_05)
E(ii,:)=[min(error_05(ii,:)), max(psnr_list_05(ii,:)), max(ssim_list_05(ii,:))];

end

%error_001 = 
if n3==1
    results_all=[A; B; C; D; E]
else
    results_all=[mean(A); mean(B); mean(C); mean(D); mean(E)]
end

% figure, imshow(beta_01)
% figure, imshow(beta_02)
% figure, imshow(beta_03)
% figure, imshow(beta_04)
% figure, imshow(beta_05)

figure;
semilogy(mean(error_01),'go-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(error_02),'b+-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(error_03),'m*-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter),hold on;
semilogy(mean(error_04),'kx-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); hold on
semilogy(mean(error_05),'rd-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); 
xlabel('Iterations');
ylabel('Tol');
%xlim([1,300]);
%ylim([1e-4, 1.1e4])
legend('ISTA', 'FISTA', 'EFISTA', 'EOPGM' ,'EOptISTA');%, 'NFISTA');
set(gca,'Fontsize',20)


figure;
semilogy(mean(psnr_list_01),'go-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(psnr_list_02),'b+-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter), hold on;
semilogy(mean(psnr_list_03),'m*-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter),hold on;
semilogy(mean(psnr_list_04),'kx-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); hold on
semilogy(mean(psnr_list_05),'rd-','LineWidth',1.5, 'MarkerIndices', 1:max_iter/10:max_iter); 
xlabel('Iterations');
ylabel('PSNR');
%xlim([1,300]);
%ylim([20, 31])
legend('ISTA', 'FISTA', 'EFISTA', 'EOPGM' ,'EOptISTA');%, 'NFISTA');
set(gca,'Fontsize',20)
