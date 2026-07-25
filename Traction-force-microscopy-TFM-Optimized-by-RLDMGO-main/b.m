clear; close all;
warning('off','all');

% 初始化表格：记录文件名、DL误差/时间、FTTC误差/时间
TError = table('Size',[0 5], 'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames',{'File_ID','DL_Error','DL_Time','FTTC_Error','FTTC_Time'});

noise = 0.0188;       % 噪声强度（根据需求调整）
L = 3e-10;       % FTTC正则化参数（根据实验调整）
E = 10670;       % 杨氏模量（单位：Pa）

for p = 1:55
    % ================= 加载104x104数据 =================
    try
        % 加载牵引力真实值（104x104）
%         fn = sprintf('C:/Users/光/Desktop/DL-TFM-main/test/sizeScale/testData/trac104/MLData00%02d.mat', p);
        fn = sprintf('C:/Users/11066/Desktop/DL-TFM-main1/test/sizeScale/testData/trac104/MLData00%02d.mat', p);
        tData104 = load(fn);
    
        
        % 加载位移场（104x104）
%         fn = sprintf('C:/Users/光/Desktop/DL-TFM-main/test/sizeScale/testData/dspl104/MLData00%02d.mat', p);
        fn = sprintf('C:/Users/11066/Desktop/DL-TFM-main1/test/sizeScale/testData/dspl104/MLData00%02d.mat', p);
        sData104 = load(fn);
    catch
        warning('文件缺失: p=%d', p);
        continue; 
    end
    


    % ================= 处理104x104数据 =================
    dspl = sData104.dspl;
    tracGT = tData104.trac;
    brdx = tData104.brdx;
    brdy = tData104.brdy;
    
    % --- 神经网络预测 ---
    tic;
    trac_DL = predictTrac(dspl, E); 
    time_DL = toc;
    err_DL = errorTrac(trac_DL, tracGT, brdx, brdy);
    
    % --- FTTC预测 ---
    tic;
    trac_FTTC = predictTracFTTC(dspl, E, L);
    time_FTTC = toc;
    err_FTTC = errorTrac(trac_FTTC, tracGT, brdx, brdy);
    
    % 记录结果
    newRow = {sprintf('#104_%02d', p), err_DL, time_DL, err_FTTC, time_FTTC};
    TError = [TError; newRow];
    
    % 打印进度
    fprintf('Processed 104x104: p=%d, DL Err=%.4f, FTTC Err=%.4f\n', p, err_DL, err_FTTC);
end














% % 处理路径2的数据（14个文件）
% for i = 1:14
%     % ================= 加载数据 =================
%     try
%         fn = sprintf('C:/Users/11066/Desktop/DL-TFM-main1/cells1/cells/dspl/MLData%03d.mat', i);
%         fileData = load(fn);  
%         dsplGT = fileData.dspl;
%         brdx = fileData.brdx;
%         brdy = fileData.brdy; 
%     catch
%         warning('文件缺失: i=%d', i);
%         continue;
%     end
%     
%     % ================= 添加噪声 =================
%     dspl_noisy = addNoise(dsplGT, noise);
%     
%     % --- 神经网络预测 ---
%     tic;
%     trac_DL = predictTrac(dspl_noisy, E);
%     time_DL = toc;
%     dspl_pred_DL = calcDspl(trac_DL, E, brdx, brdy);
%     err_DL = errorDspl(dspl_pred_DL, dsplGT, brdx, brdy, 0);  % 无缓冲区
%     
%     % --- FTTC预测 ---
%     tic;
%     trac_FTTC = predictTracFTTC(dspl_noisy, E, L);
%     time_FTTC = toc;
%     dspl_pred_FTTC = calcDspl(trac_FTTC, E, brdx, brdy);
%     err_FTTC = errorDspl(dspl_pred_FTTC, dsplGT, brdx, brdy, 10); % 缓冲区=10
%     
%     % 记录结果
%     newRow = {sprintf('#cell_%03d', i), err_DL, time_DL, err_FTTC, time_FTTC};
%     TError = [TError; newRow];
%     
%     % 打印进度
%     fprintf('Processed Cell Data: i=%d, DL Err=%.4f, FTTC Err=%.4f\n', i, err_DL, err_FTTC);
% end















%% 输出统计结果
disp('=== 汇总结果 ===');
if ~isempty(TError)
    disp('归一化误差:');
    fprintf('  神经网络 (DL): %.4f ± %.4f\n', mean(TError.DL_Error), std(TError.DL_Error));
    fprintf('  FTTC:          %.4f ± %.4f\n', mean(TError.FTTC_Error), std(TError.FTTC_Error));
    disp('平均耗时:');
    fprintf('  神经网络 (DL): %.4f s ± %.4f\n', mean(TError.DL_Time), std(TError.DL_Time));
    fprintf('  FTTC:          %.4f s ± %.4f\n', mean(TError.FTTC_Time), std(TError.FTTC_Time));
else
    disp('未找到有效数据！');
end





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

fn = sprintf('train/tracnet%d.mat',S);
% fn = sprintf('train/bestTracNet%d.mat',S);
netmat = load(fn);
net = netmat.net;
tic; 
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
function dspl = calcDspl(trac,E,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dspl = calcDspl(trac,E,brdx,brdy)
%
% Description:
%   calculate displacement field from a stress field
%
% Input:
%   trac = stress field as a SxSx2 tensor 
%   E = Young's modulus of the hydrogel in Pascals
%   brdx,brdy (optional): x and y coordinates of the cell border
%
% Output: strain field
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parse input and set up parameters
warning('off','all');
if nargin == 4
    brdx = varargin{1};
    brdy = varargin{2};
elseif nargin ~= 2
    disp("Error");
    exit;
end

S = size(trac,1); % size of the image in pixels
N=S-1; %odd number for matrix size
R=S/2; %matrix rad
v = 0.45; % Possion ratio

%% force traction stress outside the cell to be zero
if nargin == 4
   pgn = polyshape(brdx,brdy);
   [ydim,xdim,~] = size(trac);
   [Y,X] = meshgrid(1:S,1:S);
   interior = isinterior(pgn,X(:),Y(:)); 
   interior = reshape(interior,ydim,xdim);
   
   tracIX = trac(:,:,1).*interior;
   tracIY = trac(:,:,2).*interior;
else
   tracIX = trac(:,:,1);
   tracIY = trac(:,:,2);
end

%% construct Boussinesq matrices
G11 = zeros(N,N);
G12 = zeros(N,N);
G21 = zeros(N,N);
G22 = zeros(N,N);
for i=1:N
    for j=1:N
        r = sqrt((i-R)^2 + (j-R)^2);
        G11(i,j) = (1 + v)/(pi*E) * ((1-v)/r + v*(i-R)^2/r^3);
        G12(i,j) = (1 + v)/(pi*E) * (v*(i-R)*(j-R)/r^3);
        G21(i,j) = G12(i,j);
        G22(i,j) = (1 + v)/(pi*E) * ((1-v)/r + v*(j-R)^2/r^3);
    end
end
% singular value calibrated to minimize the error using FTTC plgin of
% imageJ
G11(R,R) = 0.117*10/E;
G12(R,R) = 0;
G21(R,R) = 0;
G22(R,R) = G11(R,R);

%% calculate strain field dspl from traction stress field trac
dspl = zeros(S,S,2);
% define the matrices for Boussinesq equations using combined convolutions
D11 = conv2(tracIX,G11,'same');
D21 = conv2(tracIY,G21,'same');
D12 = conv2(tracIX,G12,'same');
D22 = conv2(tracIY,G22,'same');
dspl(:,:,1)=D11+D21; 
dspl(:,:,2)=D12+D22; 
end
function errorD = errorDspl(dspl,dsplGT,brdx,brdy,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% errorS = errorStrn(strn,strnGT,brdx,brdy,buffer)
%
% Description:
%   calculate error of displacements (calculated from traction stress) 
%   relative to the ground truth (measured)
%
% Input:
%   dspl: (calculated) displacement field
%   dsplGT: (measured) ground truth displacement field
%   brdx,brdy: x and y coordinates of the cell border
%   buffer (optional): distance around the cell to be included in the 
%      calculation
%
% Output: 
%   error against ground truth, as root mean squared error
%   divided by root mean squared magnitude
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define the region for calculation
warning('off','all');
if nargin == 5
    buffer = varargin{1};
else
    buffer = 0;
end

pgn = polyshape(brdx,brdy);
pgn = polybuffer(pgn,buffer); 
[ydim,xdim,~] = size(dspl);
[Y,X] = meshgrid(1:xdim,1:ydim);
interior = isinterior(pgn,X(:),Y(:));
interior = reshape(interior,ydim,xdim);
dspl(:,:,1) = dspl(:,:,1).*interior;
dspl(:,:,2) = dspl(:,:,2).*interior;

dspl(:,:,1) = dspl(:,:,1).*interior;
dspl(:,:,2) = dspl(:,:,2).*interior;
dsplGT(:,:,1) = dsplGT(:,:,1).*interior;
dsplGT(:,:,2) = dsplGT(:,:,2).*interior;

dsplMag = sqrt(dspl(:,:,1).^2+dspl(:,:,2).^2);
dsplGTMag = sqrt(dsplGT(:,:,1).^2+dsplGT(:,:,2).^2);

mse = sum((dspl(:,:,1)-dsplGT(:,:,1)).^2+(dspl(:,:,2)-dsplGT(:,:,2)).^2,'all')/numel(dsplMag);
rmse = sqrt(mse); 
msm = sum(dsplGT(:,:,1).^2+dsplGT(:,:,2).^2,'all')/numel(dsplGTMag);
rmsm = sqrt(msm);
errorD = rmse/rmsm;
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
% This program (regularized fourier transform traction force
% reconstruction) was produced at the University of Heidelberg, BIOMS
% group of Ulrich Schwarz. It calculates traction from a gel displacement
% field.
%
% Benedikt Sabass 13-10-2008
%
% Copyright (C) 2019, Danuser Lab - UTSouthwestern 
%
% This file is part of TFM_Package.
% 
% TFM_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TFM_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TFM_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
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

function sorted = sort_nat(c)
    % sort_nat: 自然排序字符串单元格数组，例如 MLData1.mat, MLData2.mat, ..., MLData10.mat
    % c: 输入字符串单元格数组
    expr = '\d+';
    nums = cellfun(@(x) sscanf(regexp(x, expr, 'match', 'once'), '%d'), c);
    [~, idx] = sort(nums);
    sorted = c(idx);
end
