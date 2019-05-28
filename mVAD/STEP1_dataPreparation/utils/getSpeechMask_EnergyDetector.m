function [ mask_out ] = getSpeechMask_EnergyDetector( spcgIn )
    S = spcgIn;
    
    fullyVoiced = abs(S);
    SE = sum(abs(S).^2);
    MMM = median(SE);        
    maska = zeros(size(fullyVoiced));
    XX = sort(SE);
    silThr = 0.5*( XX( 1+floor(0.1*size(SE,2)) ) + MMM );
    SEThr = SE>silThr;
    
    for I = size(SEThr,2)-2:-1:1
        if SEThr(I) == 1 && SEThr(I+1) == 0
            SEThr(I+1) = 1;
            if SEThr(I+2) < SEThr(I+1), SEThr(I+2) = 1; end            
        end      
    end    
    for I = 3:size(SEThr,2)
        if SEThr(I) == 1 && SEThr(I-1) == 0
            SEThr(I-1) = 1;
            if SEThr(I-2) < SEThr(I-1), SEThr(I-2) = 1; end            
        end      
    end
    
    for I = 5:size(SEThr,2)-10
        if isequal(SEThr(I:I+2),[1 0 1]), SEThr(I+1)=1; end;
        if isequal(SEThr(I:I+3),[1 0 0 1]), SEThr(I+1)=1; SEThr(I+2)=1; end;
        if isequal(SEThr(I:I+4),[1 0 0 0 1]), SEThr(I+1)=1; SEThr(I+2)=1; SEThr(I+3)=1;  end;
    end
        
    fLims = [3,40,90,250];
    percentage = [0.5,0.5,0.3];
    hardThresholds = [1e5;1e5;1e5];
    
    for fv = 1:size(fullyVoiced,2)
        localMask = zeros(257,1);
        if SEThr(fv) == 1
            in1 = fullyVoiced(fLims(1)   : fLims(2)+2,fv);
            in2 = fullyVoiced(fLims(2)-3 : fLims(3),fv);
            in3 = fullyVoiced(fLims(3)-3 : fLims(4),fv);            

            localMask(fLims(1):fLims(2)+2)   = getLocalMask(in1,percentage(1),hardThresholds(1));
            localMask(fLims(2)-3 : fLims(3)) = getLocalMask(in2,percentage(2),hardThresholds(2));
            localMask(fLims(3)-3 : fLims(4)) = getLocalMask(in3,percentage(3),hardThresholds(3));

            maska(:,fv) =  maska(:,fv) + localMask;
        end

    end
    maska(maska>0.5) = 1;
    mask_out = maska;
end

function [ threshEstimation ] = getLocalMask(observations,percentage,hardThreshold)
    maska = zeros(size(observations));
%     choose local maxima
    nLM = ceil((size(observations,1)*percentage)/3);    
    LMs = zeros(size(observations,1),1);
    for x = 2:(size(observations,1)-1)
        if (observations(x) >= observations(x-1)) && (observations(x) >= observations(x+1))
            LMs(x)= observations(x);
        end
    end    

%     get sure silence level (nejmenších 20% hodnot je hranice)
    pom = sort(observations,1);
    silence = pom(ceil(0.2*size(observations,1)));
    threshold = min(silence,hardThreshold);
    
%     crowl local maxima
    [B,I]=sort(LMs,1,'descend');
    for mi = 1:nLM
        maska(I(mi)) = 1;
        val = LMs(I(mi));
        if val > threshold
            ls = observations(I(mi)-1); rs = I(mi+1);
            if ls > threshold
                maska(I(mi)-1) = 1; 
            else
                ls = val; 
            end;
            if rs > threshold
                maska(I(mi)+1) = 1; 
            else
                rs = val; 
            end
            
            for x = I(mi)-2:-1:1
                if x > 0 && observations(x) > rs && observations(x) < observations(x+1)
                    maska(x) = 1;
                else
                    break;
                end                
            end
            for x = I(mi)+2:size(observations,1)
                if x < size(observations,1) && observations(x) > ls && observations(x) < observations(x-1)
                    maska(x) = 1;
                else
                    break;
                end   
            end
            
        end
        
    end
    threshEstimation = maska;
end
