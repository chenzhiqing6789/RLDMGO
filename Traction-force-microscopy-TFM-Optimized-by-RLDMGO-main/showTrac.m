cellN = 4;
fn = sprintf('C:/Users/光/Desktop/DL-TFM-main/cells1/cells/dspl/MLData%03d.mat',cellN);
fileData = load(fn);
dspl = fileData.dspl;
brdx = fileData.brdx;
brdy = fileData.brdy;

trac = predictTrac(dspl,10670);
tracFilt = filtTrac(trac,trac,brdx,brdy,95);

plotTrac(tracFilt);
dsplPred = calcDspl(tracFilt,10670,brdx,brdy);
% plotError(dspl,dsplPred,brdx,brdy,95);
plotError(dspl,dsplPred,brdx,brdy);

%plotError(dsplPred,dspl,brdx,brdy);
%errorD = errorDspl(dsplPred, dspl, brdx, brdy);  % 使用0像素的缓冲区域
errorD = errorDspl(dspl, dsplPred, brdx, brdy);  % 使用0像素的缓冲区域
disp(['归一化误差: ', num2str(errorD)]);




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

fn = sprintf('train/tracnet%d-CBAM01100.mat',S);
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
function tracFilt = filtTrac(trac,tracGT,brdx,brdy,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tracFilt=filtTrac(trac,tracGT,brdx,brdy,buffer,cutoff)
%
% Description:
%   filter traction vectors, first determine the threshold magnitude
%   based on the cutoff percentile and ground truth, then remove all the 
%   extracellular stress vectors as well as intracellular vectors that are 
%   smaller in magnitude than the threshold
%
% Input:
%   trac: traction stress field to filter
%   tracGT: ground truth traction field, for determining the threshold
%   brdx,brdy: x and y coordinates of the cell border 
%   buffer (optional): buffer zone around the cell treated like interior,
%      default = 10
%   cutoff (optional): threshold for removal, 95 means keeping top 5
%      perdentile, 0 means keeping all intracellular vectors
%
% Output:
%   tracFilt: filtered traction stress field
%
% Notes: keep also vectors where either traction or ground truth is larger 
%   than the threshold 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parsing input
warning('off','all');
buffer = 10;
cutoff = 0;
if nargin==6
    buffer = varargin{1};
    cutoff = varargin{2};
elseif nargin==5
    buffer = varargin{1};
end

[ydim,xdim,~] = size(trac);
[X,Y] = meshgrid(1:xdim,1:ydim);
pgnS = polyshape(brdx,brdy);
pgnL = polybuffer(pgnS,buffer);
pgnFar = polybuffer(pgnS,buffer);
pgnFar = xor(pgnFar,polyshape([1 1 xdim xdim],[ydim 1 1 ydim])); 
far = isinterior(pgnFar,X(:),Y(:));
tracFarX = trac(:,:,1);
tracFarX = tracFarX(far);
tracFarY = trac(:,:,2);
tracFarY = tracFarY(far);
tracFarMag = mean(sqrt(tracFarX.^2+tracFarY.^2));

interior = isinterior(pgnL,Y(:),X(:));
interior = reshape(interior,ydim,xdim);

% nullify traction stresses outside the border
trac(:,:,1) = trac(:,:,1).*interior;
trac(:,:,2) = trac(:,:,2).*interior;
tracGT(:,:,1) = tracGT(:,:,1).*interior;
tracGT(:,:,2) = tracGT(:,:,2).*interior;

% generate maps of magnitude
tracMag = sqrt(trac(:,:,1).^2+trac(:,:,2).^2);
tracGTMag = sqrt(tracGT(:,:,1).^2+tracGT(:,:,2).^2);

% remove vectors smaller than those far away from the cell
tracX = trac(:,:,1).*(tracMag>tracFarMag);
tracY = trac(:,:,2).*(tracMag>tracFarMag);

if cutoff>0
    threshold = prctile(tracGTMag(:),cutoff);
    I = tracMag>=threshold | tracGTMag>=threshold;
    trac(:,:,1) = tracX.*I;
    trac(:,:,2) = tracY.*I;
end

tracFilt(:,:,1)=tracX;
tracFilt(:,:,2)=tracY;
end

function plotTrac(trac,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function plotTrac(trac,brdx,brdy,scale,thinning)
%
% Description:
%   render traction stress field as both quiver vector plot and heat map
%
% Input:
%   trac: traction stress field
%   brdx (optional): x coordinates of the border to be superimposed on the plot
%   brdy (optional): y coordinates of the border to be superimposed on the plot
%   scale (optional): relative length of quiver vectors, default = 0.002
%   thinning (optional): number of quiver vectors to skip to avoid over 
%      crowding, default = 0
%
% Output:
%  figures of quiver plot and heat map (pseudocolor rendering of the 
%  magnitude)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parsing input
if nargin==1
    scale = 0.002; 
    thinning = 1; % plot every n vectors in quiver plot
elseif nargin==5
    brdx = varargin{1};
    brdy = varargin{2};    
    scale = varargin{3};
    thinning = varargin{4}+1;
elseif nargin==3
    if numel(varargin{1})==1 && numel(varargin{2})==1
        scale = varargin{1};
        thinning = varargin{2}+1;
    else
        brdx = varargin{1};
        brdy = varargin{2};
        scale = 0.002;
        thinning = 1;
    end
else
    disp("Error");
    exit;
end

% to modify or skip (scalebar = 0) as appropriate
scalebar = 4000;
scalebarx = 5;
scalebary = 12;

color = [0 0.5 0.5];
qvwidth = 1; 
brdwidth = 1; 

%% data preparation
mag = sqrt(trac(:,:,1).^2+trac(:,:,2).^2);
if scalebar>0
   trac(scalebarx,scalebary,1) = scalebar;
   trac(scalebarx,scalebary,2) = 0;
end

[J,I] = meshgrid(1:size(trac,2),1:size(trac,1));
indx = ceil(1:thinning:numel(I));
D1 = trac(:,:,1);
D1 = D1(indx);
I = I(indx);

indx = ceil(1:thinning:numel(J));
D2 = trac(:,:,2);
D2 = D2(indx);
J = J(indx);

%% draw quiver vector map of stresses
figure('name','Stress Vectors');
q = quiver(I,J,D1*scale,D2*scale,0);
q.Color = color;
q.LineWidth = qvwidth;
xlim([0 size(trac,1)]);
xticks(0:10:size(trac,1));
ylim([0 size(trac,2)]);
yticks(0:10:size(trac,2));
if scalebar>0
    lgd = sprintf('%d Pascals',scalebar);
    text(5,5,lgd);
end
if exist('brdx','var') && exist('brdy','var')
    hold on
    plot(brdx,brdy,'--b','LineWidth',brdwidth);
    hold off;
end
daspect([1 1 1]);

%% draw heat map of stress magnitudes
figure('name','Stress Heat Map');
pc = pcolor(mag');
pc.LineStyle = 'none';
pc.FaceColor = 'interp';
xlim([0 size(trac,1)]);
xticks(0:10:size(trac,1));
ylim([0 size(trac,2)]);
yticks(0:10:size(trac,2));
colormap(jet)
colorbar;
if exist('brdx','var') && exist('brdy','var')
    hold on
    plot(brdx, brdy, '--w','LineWidth',brdwidth);
    hold off;
end
daspect([1 1 1]);
end

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

function plotError(fld,fldGT,brdx,brdy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotError(fld,fldGT,brdx,brdy)
%
% Description:
%   render the error between a measureed field and ground truth as heat
%   map
%
% Input:
%   fld: measured field
%   fldGT: ground truth field
%   brdx,brdy: x and y coordinates of the cell border 
%
% Output:
%   heat map of the magnitude of error vectors within the cell border
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% data preparation
warning('off','all');

pgn = polyshape(brdx,brdy);
[ydim,xdim,~] = size(fld);
[Y,X] = meshgrid(1:xdim,1:ydim);
interior = isinterior(pgn,X(:),Y(:));
interior = reshape(interior,ydim,xdim);

fldX = fld(:,:,1);
fldGTX = fldGT(:,:,1);
fldX = fldX.*interior;
fldGTX = fldGTX.*interior;

fldY = fld(:,:,2);
fldGTY = fldGT(:,:,2);
fldY = fldY.*interior;
fldGTY = fldGTY.*interior;

%fldMag = (sqrt(fldX.^2+fldY.^2)+sqrt(fldGTX.^2+fldGTY.^2))/2;
fldGTMag = sqrt(fldGTX.^2+fldGTY.^2);
errorMag = sqrt((fldX-fldGTX).^2+(fldY-fldGTY).^2);

% normalize by the magnitude of the field
fldErr = errorMag./fldGTMag;
fldErr(isnan(fldErr)|isinf(fldErr)) = 0;

%% render heat map
figure;
pc = pcolor(fldErr');
pc.LineStyle = 'none';
pc.FaceColor = 'interp';
xlim([0 xdim]);
xticks(0:10:xdim);
ylim([0 ydim]);
yticks(0:10:ydim);

%colormap(parula)
colormap(jet)
colorbar;
hold on

plot(brdx,brdy,'--w');
hold off
daspect([1 1 1]);
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


