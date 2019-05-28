function IRs = prepIRs( embPth , annotations )

    irs = containers.Map;
    mat = json2mat(annotations);

    nchan = 6;    
    wlen_sub=256; % STFT window length in samples
    blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
    ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
    wlen_add=1024; % STFT window length in samples for speaker localization
    del=-3; % minimum delay (0 for a causal filter)    
    
    for utt_ind=1:length(mat)
        iname = mat{utt_ind}.ir_wavfile;
        
        klic = iname;
        
        tf = isKey(irs,klic);
        if tf == 0
            ibeg=round(mat{utt_ind}.ir_start*16000)+1;
            iend=round(mat{utt_ind}.ir_end*16000);
            nbeg=round(mat{utt_ind}.noise_start*16000)+1;
            nend=round(mat{utt_ind}.noise_end*16000);

            [r,fs]=audioread([embPth iname '.CH0.wav'],[ibeg iend]);
            x=zeros(iend-ibeg+1,nchan);
            for c=1:nchan,
                x(:,c)=audioread([embPth iname '.CH' int2str(c) '.wav'],[ibeg iend]);
            end

            R=stft_multi(r.',wlen_sub);
            X=stft_multi(x.',wlen_sub);

            A=estimate_ir(R,X,blen_sub,ntap_sub,del);            
            
            irs(klic) = A;
        end
    end
IRs = irs;
end

