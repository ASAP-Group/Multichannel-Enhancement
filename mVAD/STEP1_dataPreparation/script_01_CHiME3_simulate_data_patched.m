function CHiME3_simulate_data(official)

% This script is derived from original CHiME4_simulate_data.m script and its' corresponding utils.
% The script prepares spectrograms of noise, clear speech and mixture which are saved to path specified in "myPth".
% "chimeDataPath" specifies the location of CHiME data folder.

addpath utils;
chimeDataPath = 'H:\CHiME4\data\';
upath = [chimeDataPath,'audio\16kHz\isolated\']; % path to segmented utterances
cpath = [chimeDataPath,'audio\16kHz\embedded\']; % path to continuous recordings
bpath = [chimeDataPath,'audio\16kHz\backgrounds\']; % path to noise backgrounds
apath = [chimeDataPath,'\annotations\']; % path to JSON annotations

myPth = 'E:\VAD_export\STEP1_dataPreparation\mats\'; % path for saving .mat files
nchan = 6;

% Define hyper-parameters
wlen_sub=256; % STFT window length in samples
blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
wlen_add=1024; % STFT window length in samples for speaker localization
del=-3; % minimum delay (0 for a causal filter)

%% Create simulated training dataset from original WSJ0 data %%

load('equal_filter.mat');
% Read official annotations
mat=json2mat([apath 'tr05_simu.json']);

myNfft = 512; myOverlap = myNfft - 128;
myWindow = blackman(myNfft); myFs = 16000;

% Loop over utterances
for utt_ind=1:length(mat) % this range may be changed for paralelization
    disp( [int2str(utt_ind) , ' of ' , int2str(length(mat))] );
    
    oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
    iname=mat{utt_ind}.ir_wavfile;
    nname=mat{utt_ind}.noise_wavfile;

    ibeg=round(mat{utt_ind}.ir_start*16000)+1;
    iend=round(mat{utt_ind}.ir_end*16000);
    nbeg=round(mat{utt_ind}.noise_start*16000)+1;
    nend=round(mat{utt_ind}.noise_end*16000);

    % Load WAV files
    o=audioread([upath 'tr05_org\' oname '.wav']);
    [r,fs]=audioread([cpath iname '.CH0.wav'],[ibeg iend]);
    x=zeros(iend-ibeg+1,nchan);
    n=zeros(nend-nbeg+1,nchan);
    for c=1:nchan,
        x(:,c)=audioread([cpath iname '.CH' int2str(c) '.wav'],[ibeg iend]);
        n(:,c)=audioread([bpath nname '.CH' int2str(c) '.wav'],[nbeg nend]);
    end
    
    % Compute the STFT (short window)
    O=stft_multi(o.',wlen_sub);
    R=stft_multi(r.',wlen_sub);
    X=stft_multi(x.',wlen_sub);

    % Estimate 88 ms impulse responses on 250 ms time blocks
    A=estimate_ir(R,X,blen_sub,ntap_sub,del);

    % Derive SNR
    Y=apply_ir(A,R,del);
    y=istft_multi(Y,iend-ibeg+1).';
    SNR=sum(sum(y.^2))/sum(sum((x-y).^2));
    
    % Equalize microphone
    [~,nfram]=size(O);
    O=O.*repmat(equal_filter,[1 nfram]);
    o=istft_multi(O,nend-nbeg+1).';
    
    % Compute the STFT (long window)
    O=stft_multi(o.',wlen_add);
    X=stft_multi(x.',wlen_add);
    [nbin,nfram] = size(O);
    
    % Localize and track the speaker
    X(2) = zeros(size(X(2)));
    [~,TDOAx]=localize(X);
      
    % Interpolate the spatial position over the duration of clean speech
    TDOA=zeros(nchan,nfram);
    for c=1:nchan,
        TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
    end
    
    % Filter clean speech
    Ysimu=zeros(nbin,nfram,nchan);
    for f=1:nbin,
        for t=1:nfram,
            Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
            Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
        end
    end
    ysimu=istft_multi(Ysimu,nend-nbeg+1).';

    % Normalize level and add
    ysimu=sqrt(SNR/sum(sum(ysimu.^2))*sum(sum(n.^2)))*ysimu;
    xsimu=ysimu+n;
    
    % Write WAV file
    for c=1:nchan,
        if c ~= 2
            mixture = xsimu(:,c);
            noise = n(:, c);
            speech = ysimu(:, c);
                      
            MIX = spectrogram(mixture,myWindow,myOverlap,myNfft,myFs);
            NOI = spectrogram(noise  ,myWindow,myOverlap,myNfft,myFs);
            SPE = spectrogram(speech ,myWindow,myOverlap,myNfft,myFs);
            
            SNR = 10*log10( abs(SPE).^2 ./ abs(NOI).^2 );            
            sMask = getSpeechMask_EnergyDetector(abs(SPE));
            
            save([myPth, oname , '_CH',int2str(c) , '.mat' ],'MIX','SNR','sMask','mixture');


        end
    end
end