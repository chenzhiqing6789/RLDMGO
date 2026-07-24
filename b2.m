clear; close all;
warning('off','all');

% 初始化表格：记录文件名、DL误差/时间、FTTC误差/时间
TError = table('Size',[0 3], 'VariableTypes', {'string','double','double'}, ...
    'VariableNames',{'File_ID','DL_Error','DL_Time'});

noise = 0.0188;       % 噪声强度（根据需求调整）
L = 3e-10;       % FTTC正则化参数（根据实验调整）
E = 10670;       % 杨氏模量（单位：Pa）

% 处理路径1的数据（55个文件）
 for p = 1:55
    % ================= 加载104x104数据 =================
    try

      fn = sprintf('C:/Users/admin/Desktop/DL-TFM-main1/test/generic/testData104/trac/MLData00%02d.mat', p);
%      fn = sprintf('C:/Users/admin/Desktop/DL-TFM-main1/test/sizeScale/testData/trac104/MLData00%02d.mat', p);
        tData104 = load(fn);
    
%          fn = sprintf('C:/Users/admin/Desktop/DL-TFM-main1/test/sizeScale/testData/dspl104/MLData00%02d.mat', p);
        fn = sprintf('C:/Users/admin/Desktop/DL-TFM-main1/test/generic/testData104/dspl/MLData00%02d.mat', p);
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
  
    % 记录结果
    newRow = {sprintf('#104_%02d', p), err_DL, time_DL};
    TError = [TError; newRow];
    
    % 打印进度
    fprintf('Processed 104x104: p=%d, DL Err=%.4f', p, err_DL);
 end

%% 输出统计结果
disp('=== 汇总结果 ===');
if ~isempty(TError)
    disp('归一化误差:');
    fprintf('  神经网络 (DL): %.4f ± %.4f\n', mean(TError.DL_Error), std(TError.DL_Error));
    
    disp('平均耗时:');
    fprintf('  神经网络 (DL): %.4f s ± %.4f\n', mean(TError.DL_Time), std(TError.DL_Time));
 
else
    disp('未找到有效数据！');
end


function errorT = errorTrac(trac,tracGT,brdx,brdy,varargin)


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
%%
S = length(dspl);
mag = S/104;

conversion = E/10*mag; % convert from Newons/pixel to Pa

fn = sprintf('train/bestTracNet%d.mat',S);
netmat = load(fn);
net = netmat.bestNet;
tic; 
% predict traction stress from preloaded displacements
trac = activations(net,dspl,'output')*conversion;

end



