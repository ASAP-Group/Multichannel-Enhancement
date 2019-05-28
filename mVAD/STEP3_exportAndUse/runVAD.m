function [ masks ] = runVAD( x , NNPack )

    SGdata = spectrogram(x,NNPack('window'),NNPack('nfft')-NNPack('shift'),NNPack('nfft'),NNPack('FS'),'yaxis');
    SGData = AdaptGain(abs(SGdata),NNPack('lambda'));

    masks = zeros( 2*size(SGData,1) , size(SGData,2) );

    W_in = NNPack('W_in');
    B_in = NNPack('B_in');
    W_HL1 = NNPack('W_HL1');
    B_HL1 = NNPack('B_HL1');
    W_HL2 = NNPack('W_HL2');
    B_HL2 = NNPack('B_HL2');
    W_out = NNPack('W_out');
    B_out = NNPack('B_out');
        
    for frame = 1:size(SGdata,2)
        inputs = SGData(:,frame);
        inputs = (inputs - NNPack('MEANin')) .* NNPack('STDin');
            
        layerIns = inputs' * W_in + B_in;
        activated = layerIns .* (layerIns > 0.0); % ReLU    
        layerIns = activated * W_HL1 + B_HL1;
        activated = layerIns .* (layerIns > 0.0); % ReLU        
        layerIns = activated * W_HL2 + B_HL2;
        activated = layerIns .* (layerIns > 0.0); % ReLU
        layerIns = activated * W_out + B_out;     

        masks(:,frame) = applySigmoid( layerIns );    
    end
end

function auto = AdaptGain(SG,lambda)
    last = lambda * mean( SG(:,1).^2 );
    for i = 1:size(SG,2)
        last = lambda*last + mean( SG(:,i).^2 );  
        SG(:,i) = SG(:,i) / sqrt(last);
    end
    auto = SG;
end
function [ OUT ] = applySigmoid( IN )
    OUT = 1 ./ (ones(size(IN)) + exp(-IN));
end