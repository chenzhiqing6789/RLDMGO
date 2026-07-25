clc; clear;
TError = table({'fn'}, 0, 0, 0, 0, 'VariableNames', {'File_ID', 'Size_104_104_L_N', 'Size_256_256_L_N', 'Size_104_104_L_with_noise', 'Size_256_256_L_with_noise'});
noise = 0.01765;
n = 1;

for p = 1:55
    fn = sprintf('D:/DL-TFM-main/testDATA/test/sizeScale/testData/trac160/MLData00%02d.mat', p);
    try
        tData160 = load(fn);
    catch
        continue;
    end

    fn = sprintf('D:/DL-TFM-main/testDATA/test/sizeScale/testData/trac104/MLData00%02d.mat', p);
    tData104 = load(fn);
    fn = sprintf('D:/DL-TFM-main/testDATA/test/sizeScale/testData/trac256/MLData00%02d.mat', p);
    tData256 = load(fn);
    fn = sprintf('D:/DL-TFM-main/testDATA/test/sizeScale/testData/dspl104/MLData00%02d.mat', p);
    sData104 = load(fn);
    fn = sprintf('D:/DL-TFM-main/testDATA/test/sizeScale/testData/dspl256/MLData00%02d.mat', p);
    sData256 = load(fn);
    
    fn = sprintf('#00%02d', p);
    TError{n, 1} = {fn};
    
    % 处理104x104数据
    dspl = sData104.dspl;
    tracGT = tData104.trac;
    brdx = tData104.brdx;
    brdy = tData104.brdy;
    trac = predictTrac(dspl, 10670);
%     trac =predictTracFTTC(dspl,10670,0.000000000000001);
    err = errorTrac(trac, tracGT, brdx, brdy);
    TError{n, 2} = err;
    
    % 添加噪声并预测
    dspl = addNoise(dspl, noise);
%     trac = predictTrac(dspl, 10670);
    trac =predictTracFTTC(dspl,10670,0.000000000000001);
    err = errorTrac(trac, tracGT, brdx, brdy);
    TError{n, 4} = err;
    
    % 处理256x256数据
    dspl = sData256.dspl;
    tracGT = tData256.trac;
    brdx = tData256.brdx;
    brdy = tData256.brdy;
%     trac = predictTrac(dspl, 10670);
    trac =predictTracFTTC(dspl,10670,0.0000000000001);
    err = errorTrac(trac, tracGT, brdx, brdy);
    TError{n, 3} = err;
    
    % 添加噪声并预测
    dspl = addNoise(dspl, noise);
%     trac = predictTrac(dspl, 10670);
    trac =predictTracFTTC(dspl,10670,0.000000000000001);
    err = errorTrac(trac, tracGT, brdx, brdy);
    TError{n, 5} = err;
    
    n = n + 1;
end

disp('normalized error')
disp(mean(TError{:, 2:end}))

function trac = predictTrac(dspl,E)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trac = predictTrac(dspl,E)
%
% Description:
%   calculate traction stress field trac from strain field dspl
%
% Input:
%   dspl = displacements, as a SxSx2 tensor where the two z channels 
%      contain x and y component of the displacement respectively
%   E = Young's modulus of the substrate in Pascals
%
% Output: 
%   trac = traction stress field trac as a SxSx2 tensor where the two z
%   channels contain x and y components of the traction stress respectively
%   in Pascals
%
% Requirement: mat file tracnetS.mat in the search path, containing 
%   trained neural network for SxS fields  
%
% Note: tracnet104.mat was trained for 10 Newtons/pixel, tracnet160.mat 
%   and tracnet256.mat were trained for 10/mag Newtons/pixel, where 
%   mag = S/104 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
S = length(dspl);
mag = S/104;
conversion = E/10*mag; % convert from Newons/pixel to Pa
fn = sprintf('D:/DL-TFM-main/train/tracnet%d.mat',S);
netmat = load(fn);
net = netmat.net;

% predict traction stress from preloaded displacements
trac = activations(net,dspl,'output')*conversion;

end

%% explanation of conversion
%   conversion factor from Newtons/pixel to Pascals = (pix/1e6)^2
%   where pix is the width/length of the square pixel in microns
%   E0 for the training set = 10/mag*(1e6/pix)^2 Pascals
%   Conversion for a different Young's modulus E = E/Eo = 
%   E/mag/(10/mag*(1e6/pix)^2)
%   Raw output is in Newtons/pix, conversion to Pascals = 
%   E/(10/mag*(1e6/pix)^2) * (pix/1e6)^2 = E/10*mag
function errorT = errorTrac(trac,tracGT,brdx,brdy,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% errorT = errorTrac(trac,tracGT,brdx,brdy,cutoff)
%
% Description:
%   calculate error of traction stress field relative to the ground truth, 
%   as normalized root mean squared error, for cell interior only;
%   if cutoff > 0 then calculate only the error of vectors > cutoff, 
%   including positions where only ground truth is > cutoff
%
% Input:
%   trac: traction stress field
%   tracGT: ground truth traction stress field
%   brdx,brdy: x and y coordinates of the cell border
%   cutoff(optional); cutoff percentile for inclusion, e.g. 95 means 
%      caloulating only the error of top 5% vectors 
%
% Output: 
%   error against ground truth, as root mean squared error
%   divided by root mean squared magnitude
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define the region within the cell for calculating the error
warning('off','all');
if nargin==4
    cutoff = 0;
elseif nargin==5
    cutoff = varargin{1};
else
    disp('Error')
end
pgn = polyshape(brdx,brdy);
[ydim,xdim,~] = size(trac);
[X,Y] = meshgrid(1:xdim,1:ydim);
interior = isinterior(pgn,Y(:),X(:));
interior = reshape(interior,ydim,xdim);

trac(:,:,1) = trac(:,:,1).*interior;
trac(:,:,2) = trac(:,:,2).*interior;
tracGT(:,:,1) = tracGT(:,:,1).*interior;
tracGT(:,:,2) = tracGT(:,:,2).*interior;

tracMag = sqrt(trac(:,:,1).^2+trac(:,:,2).^2);
tracGTMag = sqrt(tracGT(:,:,1).^2+tracGT(:,:,2).^2);

if cutoff>0
    threshold = prctile(tracGTMag(interior),cutoff);
    I = tracMag>threshold | tracGTMag>threshold;
    trac(:,:,1) = trac(:,:,1).*I;
    trac(:,:,2) = trac(:,:,2).*I;
    tracGT(:,:,1) = tracGT(:,:,1).*I;
    tracGT(:,:,2) = tracGT(:,:,2).*I;
else
    I = ones(ydim,xdim);
end

mse = sum((trac(:,:,1)-tracGT(:,:,1)).^2 + (trac(:,:,2)-tracGT(:,:,2)).^2,'all')/nnz(interior.*I);
rmse = sqrt(mse);
msm = sum(tracGT(:,:,1).^2+tracGT(:,:,2).^2,'all')/nnz(interior.*I);
rmsm = sqrt(msm);
errorT = rmse/rmsm;

end
function dsplN = addNoise(dspl,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description: add Gaussian noise to a modeled displacement field
%
% Input:
%   dspl: displacement field
%   N: magnitude of the noise
%
% Output: displacement field aster imposing the noise
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = length(dspl);
stdev = N/sqrt(2); % measured = 0.00765
rng('shuffle');
noise = random('normal',0,stdev,[S,S]);
dsplN(:,:,1) = dspl(:,:,1) + noise;
noise = random('normal',0,stdev,[S,S]);
dsplN(:,:,2) = dspl(:,:,2) + noise;
end

function trac = predictTracFTTC(dspl,E,L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% trac = predictTracFTTC(dspl,E,L)
%
% Description:
%   calculate traction stress field trac from strain field dspl using FTTC
%   a simple wrapper to use US Schwarz's program reg_fourier_TFM in a 
%   similar syntax as that for deep learning predictTrac
%
% Input:
%   dspl = displacements, as a SxSx2 tensor where the two z channels 
%      contain x and y component of the displacement respectively
%   E = Young's modulus of the substrate in Pascals
%   v = Poisson ratio of the substrate
%   L = regularization factor
%
% Output: 
%   trac = traction stress field trac as a SxSx2 tensor where the two z
%   channels contain x and y components of the traction stress respectively
%   in Pascals
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
v = 0.45;
S = length(dspl);
[x,y] = meshgrid(1:1:S,1:1:S);
reg_grid = zeros(S,S,2);
reg_grid(:,:,1) = x';
reg_grid(:,:,2) = y';

% predict traction stress from preloaded displacements
[~ ,~ ,~ , ~ ,~ , trac] = reg_fourier_TFM(reg_grid, dspl, E, v, 1, 1, S, S, L);
end

function  [pos,vec,force,fnorm,energie,f] = reg_fourier_TFM(grid_mat,u,E,s, pix_durch_my, cluster_size, i_max, j_max, L,Rx,Ry)
% added by Achim:
% Input : grid_mat, u, cluster_size have to be in the same units, namely
%         pixels. If they are given in different units e.g. meters, then a
%         different scaling factor for the elastic energy has to be
%         applied! The force value remains, however, unaffected, see below.
% Output: The output force is actually a surface stress with the same units
%         as the input E! In particular, the unit of the output force is
%         independent of the units of the input grid_mat,u and cluster_size
%         The reason for this is essentially that the elastic stress is
%         only dependent on the non-dimensional strain which is given by
%         spatial derivatives of the displacements, that is du/dx. If u and
%         dx (essentially cluster_size) are in the same units, then the
%         resulting force has the same dimension as the input E.
% updated by Sangyoon Han for usage for L1 regularization

    %nN_pro_pix_fakt = 1/(10^3*pix_durch_my^2);
    %nN_pro_my_fakt = 1/(10^3);
    
    V = 2*(1+s)/E;
    
    kx_vec = 2*pi/i_max/cluster_size.*[0:(i_max/2-1) (-i_max/2:-1)];
    ky_vec = 2*pi/j_max/cluster_size.*[0:(j_max/2-1) (-j_max/2:-1)];
    kx = repmat(kx_vec',1,j_max);
    ky = repmat(ky_vec,i_max,1);
    if nargin<10
        Rx=ones(size(kx));
        Ry=ones(size(ky));
    end

    kx(1,1) = 1;
    ky(1,1) = 1;
    
    X = i_max*cluster_size/2;
    Y = j_max*cluster_size/2; 
   
    g0x = pi.^(-1).*V.*((-1).*Y.*log((-1).*X+sqrt(X.^2+Y.^2))+Y.*log( ...
      X+sqrt(X.^2+Y.^2))+((-1)+s).*X.*(log((-1).*Y+sqrt(X.^2+Y.^2) ...
      )+(-1).*log(Y+sqrt(X.^2+Y.^2))));
    g0y = pi.^(-1).*V.*(((-1)+s).*Y.*(log((-1).*X+sqrt(X.^2+Y.^2))+( ...
      -1).*log(X+sqrt(X.^2+Y.^2)))+X.*((-1).*log((-1).*Y+sqrt( ...
      X.^2+Y.^2))+log(Y+sqrt(X.^2+Y.^2))));
    
    Ginv_xx =(kx.^2+ky.^2).^(-1/2).*V.*(kx.^2.*L.*Rx+ky.^2.*L.*Ry+V.^2).^(-1).*(kx.^2.* ...
              L.*Rx+ky.^2.*L.*Ry+((-1)+s).^2.*V.^2).^(-1).*(kx.^4.*(L.*Rx+(-1).*L.*s.*Rx)+ ...
              kx.^2.*((-1).*ky.^2.*L.*Ry.*((-2)+s)+(-1).*((-1)+s).*V.^2)+ky.^2.*( ...
              ky.^2.*L.*Ry+((-1)+s).^2.*V.^2));
    Ginv_yy = (kx.^2+ky.^2).^(-1/2).*V.*(kx.^2.*L+ky.^2.*L.*Ry+V.^2).^(-1).*(kx.^2.* ...
              L.*Rx+ky.^2.*L.*Ry+((-1)+s).^2.*V.^2).^(-1).*(kx.^4.*L+(-1).*ky.^2.*((-1)+ ...
              s).*(ky.^2.*L.*Rx+V.^2)+kx.^2.*((-1).*ky.^2.*L.*Ry.*((-2)+s)+((-1)+s).^2.* ...
              V.^2));
    Ginv_xy = (-1).*kx.*ky.*(kx.^2+ky.^2).^(-1/2).*s.*V.*(kx.^2.*L.*Rx+ky.^2.*L.*Ry+ ...
              V.^2).^(-1).*(kx.^2.*L.*Rx+ky.^2.*L.*Ry+((-1)+s).*V.^2).*(kx.^2.*L.*Rx+ky.^2.* ...
              L.*Ry+((-1)+s).^2.*V.^2).^(-1);


    Ginv_xx(1,1) = 1/g0x;
    Ginv_yy(1,1) = 1/g0y;
    Ginv_xy(1,1) = 0;

    Ginv_xy(i_max/2+1,:) = 0;
    Ginv_xy(:,j_max/2+1) = 0;

    Ftu(:,:,1) = fft2(u(:,:,1));
    Ftu(:,:,2) = fft2(u(:,:,2));

    Ftf(:,:,1) = Ginv_xx.*Ftu(:,:,1) + Ginv_xy.*Ftu(:,:,2);
    Ftf(:,:,2) = Ginv_xy.*Ftu(:,:,1) + Ginv_yy.*Ftu(:,:,2);

    f(:,:,1) = ifft2(Ftf(:,:,1),'symmetric');
    f(:,:,2) = ifft2(Ftf(:,:,2),'symmetric');
    
    pos(:,1) = reshape(grid_mat(:,:,1),i_max*j_max,1);
    pos(:,2) = reshape(grid_mat(:,:,2),i_max*j_max,1);

    vec(:,1) = reshape(u(:,:,1),i_max*j_max,1);
    vec(:,2) = reshape(u(:,:,2),i_max*j_max,1);

    force(:,1) = reshape(f(:,:,1),i_max*j_max,1);
    force(:,2) = reshape(f(:,:,2),i_max*j_max,1);     
   
    fnorm = (force(:,1).^2 + force(:,2).^2).^0.5;
    energie = 1/2*sum(sum(u(2:end-1,2:end-1,1).*f(2:end-1,2:end-1,1) + u(2:end-1,2:end-1,2).*f(2:end-1,2:end-1,2)))*(cluster_size)^2*pix_durch_my^3/10^6; 
end
