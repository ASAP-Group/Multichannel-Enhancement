clc; clear all; close all;

% This script computes the normalization statistics, randomly splits data into training and validation data sets and prepares binary files for NNET training process.
% "PTH" is path to mat files prepared by preceding step.
% "sharedPth" is directory where statistics will be saved.
% "binPth" is target directory for training binaries (approx. 20 GB of disk space is required and 32 GB RAM).

PTH ='E:\VAD_export\STEP1_dataPreparation\mats\'; % path where .mat files are saved
sharedPth = 'E:\VAD_export\sharedData\'; % path for saving shared data
binPth = 'E:\VAD_export\STEP1_dataPreparation\bins\'; % output path for training binary files

items = dir(PTH);

%% estimation of statistics for normalization

M=[];S=[];
prcCnt = ceil(length(items)/100);
for FI = 1:length(items)
    item = items(FI);
    if mod(FI,prcCnt) == 0
        disp(['stats ... ',int2str(FI/prcCnt),'%',' computed'])
    end
    if item.isdir() == 0
        load([PTH,'/',item.name()]);
        NM = strsplit(item.name,'.');
        data = abs(MIX);
        
        mn = mean(data,2);
        st = std(data,1,2);
        M = [M,mn];
        S = [S,st];
    end
end;

MEAN = mean(M,2);
STD  = mean(S,2);

save([sharedPth,'STATS.mat'],'STD','MEAN'); 
disp('... stats saved')


%% choosing train / test / validation datasets

RPM = randperm(length(items));
nms = repmat( cellstr('') , length(items) , 1 ); 

cntr = 1;
for FI = 1:length(items)
    item = items(RPM(FI));
    NX=strsplit(item.name(),'_');
    if item.isdir() == 0 && strcmp(NX{4},'CH1.mat') == 1
        nms(cntr) = { item.name() };
        cntr = cntr + 1;
    end
end;
nms = nms(1:cntr);

test=[]; train=[]; valid=[];
RNGs = randperm(size(nms,1));
for I = 1:length(nms)
   nm = nms{ RNGs(I) };
   if mod(I,10) == 3
       test = [test;nm];
   elseif mod(I,10) == 7
       valid = [valid;nm];
   else
       train = [train;nm];
   end;
end

ds = containers.Map();
ds('train') = train;
ds('valid') = valid;
ds('test') = test;

save([sharedPth,'dataSets.mat'],'ds');

disp('... dataSest chosen');

%% genarating binary files for Torch

addpath('utils');
mkdir(binPth);
load([sharedPth,'dataSets.mat']);
load([sharedPth,'STATS.mat']);

% matlab needs too much memory - train data are split into subbatches
DS = containers.Map();
DS('test') = ds('test');
DS('valid') = ds('valid');
T = ds('train');
DS('train_1') = T(   1:1150,:);
DS('train_2') = T(1151:2300,:);
DS('train_3') = T(2301:3450,:);
DS('train_4') = T(3450:4600,:);
DS('train_5') = T(4601:end ,:);

gkeys = keys(DS);
for G = 1:length(gkeys)
    group = gkeys{G};
    disp(['... prepairing dataset: ',group]);
        
    frameCount = 0;
    xfiles = DS(group);
    disp('... allocating memory');
    for find = 1:size(xfiles,1)
        load([PTH,xfiles(find,:)]);        
        frameCount = frameCount + size(sMask,2);
    end
    data_in  = zeros(257,frameCount);
    data_out = zeros(514,frameCount);
    
    frameCount = 1;
    for find = 1:size(xfiles,1)
        load([PTH,xfiles(find,:)]);
        
        for c = 1:size(MIX,2)
            iii = abs(MIX(:,c));
            iii = (iii-MEAN) ./ STD;
            
            ooo_s = ( SNR(:,c) > 5 ) .* sMask(:,c);
            ooo_n = ( sMask(:,c) == 0) | (SNR(:,c) < 0);
            
            data_in(:,frameCount) = iii;
            data_out(:,frameCount) = [ooo_s;ooo_n];
            frameCount = frameCount + 1;
        end        

        if mod(find,50) == 0
           disp(['    ... used ',int2str(find),' files of ',int2str(size(xfiles,1))]);
        end        
    end
        
    saveMatFile4Torch( [binPth,group,'.nin'] , data_in )
    saveMatFile4Torch( [binPth,group,'.05o'] , data_out )
end
disp('... data preparation finished');