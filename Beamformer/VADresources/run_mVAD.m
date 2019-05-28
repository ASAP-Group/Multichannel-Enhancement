function [ vad,nad ] = run_mVAD( NN_package , spcgIn )

    inputs = abs(spcgIn);
%     inputs = AdaptGain(inputs,NN_package(1).lambda);
    matOut = zeros( 2*size(inputs,1) , size(inputs,2) );
    noLayers = NN_package(1).noLayers;
    
    for frm = 1:size(inputs,2)
        inps = inputs(:,frm);
        inps = (inps - NN_package(1).MEANin) .* NN_package(1).STDin;       
        
        activated = inps(:)';
        for L = 1:noLayers-1    % ReLU-activated layers
            layerIns = activated*NN_package(L).weights + NN_package(L).biases;
            activated = layerIns .* (layerIns > 0.0);   % ReLU
        end
        
        % output layer
        matOut(:,frm) = applySigmoid( activated*NN_package(noLayers).weights + NN_package(noLayers).biases );        
    end
    
    vad = matOut(1:257,:);
    nad = matOut(258:end,:);
end

function [ OUT ] = applySigmoid( IN )
    OUT = 1 ./ (ones(size(IN)) + exp(-IN));
end
% function [ asg ] = AdaptGain(SG,lambda)
%     last = lambda * mean( SG(:,1).^2 );
%     for i = 1:size(SG,2)
%         last = lambda*last + mean( SG(:,i).^2 );  
%         SG(:,i) = SG(:,i) / sqrt(last);
%     end
%     asg = SG;
% end