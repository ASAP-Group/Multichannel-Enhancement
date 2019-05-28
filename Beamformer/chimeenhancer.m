function [out, ysc] = chimeenhancer( x , BF , VADsel , bf_settings  )
% x ... original noisy signals
% BF ... beamformer: 'FSB' or 'MVDR'
% VADsel ... selected VAD detector: 'mVAD'
% bf_settings ... structure with fields
%   - BMstructure ... the structure of blocking matrix that is determined 
%     through which RTFs are used (RTFs between 'what' channels): 
%     'neighbours' or 'onereference'
%   - postfilter ... apply postfilter: 'post2' / 'off'
%   - vnad_median ... all VAD/NAD: Median filtering of the VAD/NAD masks
%   - thrs ...  the threshold value for correlation coefficient 
%     to accept an input channel
%   - blockn ... number of frames in processing batch
%   - nsubblock ... number of frames in subblock of processing batch
%   - tchan ... preferred channel in the input data (used as reference)
%
% out ... enhanced signal
% ysc ... estimated noise on the output (before the enhancement)/ check if
%         the target is blocked well

% Author(s): Zbynìk Koldovský, Jiøí Málek, Marek Boháè
% Technical University of Liberec
% Studentská 1402/2, LIBEREC
% Czech Republic
%
%
% This is unpublished proprietary source code of TECHNICAL UNIVERSITY OF
% LIBEREC, CZECH REPUBLIC.
% 
% The purpose of this software is the dissemination of scientific work for
% scientific use. The commercial distribution or use of this source code is
% prohibited. The copyright notice does not evidence any actual or intended
% publication of this code. Term and termination:
% 
% This license shall continue for as long as you use the software. However,
% it will terminate if you fail to comply with any of its terms and
% conditions. You agree, upon termination, to discontinue using, and to
% destroy all copies of, the software.  Redistribution and use in source and
% binary forms, with or without modification, are permitted provided that
% the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer (Limitation of
% warranties and liability). Redistributions in binary form must reproduce
% the above copyright notice, this list of conditions and the following
% disclaimer (Limitation of warranties and liability) in the documentation
% and/or other materials provided with the distribution. Neither name of
% copyright holders nor the names of its contributors may be used to endorse
% or promote products derived from this software without specific prior
% written permission.
% 
% The limitations of warranties and liability set out below shall continue
% in force even after any termination.
% 
% Limitation of warranties and liability:
% 
% THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
% WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE HEREBY
% DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS  OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
% OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
% SUCH DAMAGE.  

%% verification of input setting
if(nargin < 3)
    error('!!! Beamformer requires 3 or 4 input arguments. See help for details.');
end

if(nargin == 3)
    bf_settings = [];
end

set2default = 0;

if(isfield(bf_settings,'BMstructure')) % using 1 reference channel or neighbours
   BMstructure = bf_settings.BMstructure; 
else
   BMstructure = 'onereference';
   set2default = set2default+1;
end

if(isfield(bf_settings,'postfilter')) % post-filtering
   postfilter = bf_settings.postfilter; 
else
   postfilter = 'post2';
   set2default = set2default+1;
end

if(isfield(bf_settings,'vnad_median')) % median filtering of VAD (NAD) masks
   vnad_median = bf_settings.vnad_median; 
else
   vnad_median = true;
   set2default = set2default+1;
end

if(isfield(bf_settings,'thrs')) % the threshold value for correlation coefficient to accept an input channel
   critthreshold = bf_settings.thrs; 
else
   critthreshold = 0.05;
   set2default = set2default+1;
end

if(isfield(bf_settings,'blockn')) % the length of the main batch
   blockn = bf_settings.blockn; 
else
   blockn = 100;
   set2default = set2default+1;
end 

if(isfield(bf_settings,'nsubblock')) % the number of subblocks within the RTF estimation
   nsubblock = bf_settings.nsubblock; 
else
   nsubblock = 10;
   set2default = set2default+1;
end 

if(isfield(bf_settings,'tchan')) % index of reference channel
   tchan = bf_settings.tchan; 
else
   tchan = 1;
   set2default = set2default+1;
end 

if (set2default > 0)
    warning(['!!! ',int2str(set2default),' beamformer settings were set to default. ','See help for details.']);
end

%% 
FS = 16000; % sampling frequency
epsilon = 0.001; % the constant for diagonal loading of covariance
NFFT = 512; % FFT window size
FFTSHIFT = 128; % FFT shift

m = size(x,2);

%% STFT
spcGram = spectrogram(x(:,1),hamming(NFFT),NFFT-FFTSHIFT,NFFT,FS,'yaxis');
spcGram(:,:,2:m) = 0;
for i=2:m
  spcGram(:,:,i) = spectrogram(x(:,i),hamming(NFFT),NFFT-FFTSHIFT,NFFT,FS,'yaxis');
end

[K, Nb, ~] = size(spcGram);

%% VAD
VAD = zeros(K,Nb,m);
NAD = zeros(K,Nb,m);
persistent nndata
persistent nndataType

addpath('VADResources');
if strcmp(VADsel,'mVAD')  
    if isempty(nndata) || ~strcmp(nndataType,'mVAD')
        nndata = load('mVAD_nnet.mat');    
        nndataType = 'mVAD';    
    end
    
    for lp=1:m
        [VAD(:,:,lp),NAD(:,:,lp)] = run_mVAD(nndata.NN_data, spcGram(:,:,lp));
		% NAD is not explicitely used in the script, may be replaced by ~
    end 
end
rmpath('VADResources');

if(vnad_median)
    VAD = repmat(median(VAD,3),1,1,m);
end



%% Processing

DoProcessing = true(K,1); % skip processing of frequencies that are 'false'
IN = permute(spcGram,[3 2 1]);
OUT = zeros(K, Nb);
Ysc = zeros(K, Nb);

stop = false;  % auxiliary variable, prevents from processing last short chunk of data
for i = 1:Nb   % redundant for-cycle for future variants of on-line implementations    
    if rem(i,blockn) == 1 % process new batch of data of length blockn
        
        if i+2*blockn<=Nb
            batch = i:i+blockn-1;
        else % if we are before the end (batch must be longer than blockn)
            if(stop)
                break;
            else    
                batch = i:Nb;
                stop=true;
            end
        end
        
        %% Microphone failure detection using correlation coefficients
        
        CC = corrcoef(x((i-1)*FFTSHIFT+1:(batch(end)-1)*FFTSHIFT+NFFT,:));
        crit = max(CC-eye(m));
        schan = find(crit > critthreshold); % selected channels
        mm = length(schan);
        if mm < 2
            [~, schan] = sort(crit,'descend');
            schan = schan(1:2);
            mm = 2;
        end

        if sum(schan==tchan)>0 % select the output channel within this batch, tchan is preferred
            best = find(schan==tchan); % target channel
        else
            [~, best] = max(crit(schan)); % otherwise select the channel with maximum correlation
        end
        % NOTE: 'best' is the index within the selected channels, not within
        % all input channels
        
        %% RTF estimation
        
        H = ones(K,m);
        Nsb = floor(length(batch)/nsubblock); % the number of samples within a subblock
        PLR = zeros(K,nsubblock);
        PLL = zeros(K,nsubblock);
        if strcmp(BMstructure,'onereference')
            for p = 1:mm
                 if p ~= best
                    for j = 1:nsubblock
                        sbatch = batch((j-1)*Nsb+1:j*Nsb);
                        PLR(:,j) = sum(spcGram(:,sbatch,schan(best)).*conj(spcGram(:,sbatch,schan(p))).*...
                            (VAD(:,sbatch,schan(p))),2);
                        PLL(:,j) = sum(spcGram(:,sbatch,schan(p)).*conj(spcGram(:,sbatch,schan(p))).*...
                            (VAD(:,sbatch,schan(p))),2);
                    end
                    G = zeros(K,1);
                    for k = 1:K
                        aux = pinv([PLL(k,:).' ones(nsubblock,1)])*PLR(k,:).';
                        G(k) = aux(1);
                    end
                    H(:,schan(p)) = G(1:K);
                end
            end
        elseif strcmp(BMstructure,'neighbours')
            for p = 1:mm-1
                for j = 1:nsubblock
                    sbatch = batch((j-1)*Nsb+1:j*Nsb);
                    PLR(:,j) = sum(spcGram(:,sbatch,schan(p+1)).*conj(spcGram(:,sbatch,schan(p))).*...
                        (VAD(:,sbatch,schan(p))),2);
                    PLL(:,j) = sum(spcGram(:,sbatch,schan(p)).*conj(spcGram(:,sbatch,schan(p))).*...
                        (VAD(:,sbatch,schan(p))),2);
                end
                G = zeros(K,1);
                for k = 1:K
                    aux = pinv([PLL(k,:).' ones(nsubblock,1)])*PLR(k,:).';
                    G(k) = aux(1);
                end
                H(:,schan(p)) = G(1:K);
            end            
        end

        
        %% Beamforming
        for k = 1:K
            if DoProcessing(k) 
                
                C = IN(schan,batch,k)*IN(schan,batch,k)'/length(batch); % input covariance estimation 
                AUXC = C + epsilon*diag(diag(C)); % diagonal loading

                % blocking matrix
                if strcmp(BMstructure,'onereference')
                    W = zeros(mm); 
                    for p = 1:mm
                        W(p,p)=H(k,schan(p)); 
                        W(p,best)=-1;
                    end 
                    W(best,:)=[];
                elseif strcmp(BMstructure,'neighbours')
                    W = zeros(mm-1,mm); 
                    for p = 1:mm-1
                        W(p,p)=H(k,schan(p)); 
                        W(p,p+1)=-1;
                    end
                end

                % least-squares projection of the noise subspace back to
                % microphones (briefly, least-squares noise estimation)
                AUX = W*AUXC*W';
                AUX = AUXC*W'*(AUX\W);

                if(strcmp(BF,'MVDR'))
                    Cy = AUX*AUXC; 
                end    
                AUX = AUX * IN(schan,batch,k);  


                % steering vector
                if strcmp(BF,'FSB')
                    h = H(k,schan)/mm; % valid only for 'onereference'
                elseif strcmp(BF,'MVDR')
                    Cy = Cy + trace(Cy)/2*eye(mm); 
                    a = null(W); % steering vector                   
                    w = pinv(Cy)*a;
                    h = AUXC*(w*w')/(w'*AUXC*w);
                    h = h(best,:);    
                end
                
                % same code for FSB and MVDR
                OUT(k,batch) = h * IN(schan,batch,k); % filter-and-sum output
                Ysc(k,batch) = h * AUX; % estimated noise in the FSB output (also the residual noise)

            else
                % do not process the frequency
                OUT(k,batch) = IN(schan(best),batch,k);
            end                
        end
    end
end

%% The postfilter
if(strcmp(postfilter,'post2'))
    % Wiener gain (actually valid for FSB, ad hoc for MVDR)
    GAIN = max(abs(OUT).^2-abs(Ysc).^2,ones(size(OUT))*0.0001)./(abs(OUT).^2+0.0001); % Wiener postfilter

    % Tricks with GAIN using VAD
    GAIN(mean(VAD,3) > 0.3) = 1;
    
    % select frequencies to-be postfiltered
    Kpost = 1:100; 
    
    % Other tricks with GAIN
    GAIN(:,1:3)=0.01;
    
    % A milder postfilter than in post1:
    OUT(Kpost,:) = OUT(Kpost,:).* sqrt(1/8 + 7*GAIN(Kpost,:)/8);      
end

%% inverse STFT for output
win = tuckeywin(NFFT,0.2);
ysc = istft([Ysc; flipud(Ysc(2:K-1,:))],FFTSHIFT,NFFT,win)';
out = istft([OUT; flipud(OUT(2:K-1,:))],FFTSHIFT,NFFT,win)';
end





function [x] = istft(X, wshift, NFFT, window)

    [N,M] = size(X);
    
    if(nargin<3)
        NFFT=N; 
    end
    
    if(nargin<4)
        window = boxcar(N); 
    end

    Y=real(ifft(X,NFFT,'symmetric'));
    x=zeros(1,(M-1)*wshift+NFFT);
 
    for l=0:M-1
        x(l*wshift+1:l*wshift+NFFT)=x(l*wshift+1:l*wshift+NFFT)+Y(1:NFFT,l+1)'.*window';
    end

    q=NFFT/wshift;
    for l=2:q
        x((l-1)*wshift+1:l*wshift)=x((l-1)*wshift+1:l*wshift)/l;
    end

    x(q*wshift+1:M*wshift)=x(q*wshift+1:M*wshift)/q;

    for l=M:M+q-3
        x(l*wshift+1:(l+1)*wshift)=x(l*wshift+1:(l+1)*wshift)/(M+q-1-l);
    end
end

    
function [y] = tuckeywin(L,r)

    y = ones(L,1);

    low = round((0:1/L:r/2)*L)+1;

    y(low) = (1 + cos(2*pi/r*(low/L-r/2)))/2;

    high = find((0:L-1)/L>1-r/2);

    y(high) = (1 + cos(2*pi/r*(high/L -1 + r/2)))/2;
end