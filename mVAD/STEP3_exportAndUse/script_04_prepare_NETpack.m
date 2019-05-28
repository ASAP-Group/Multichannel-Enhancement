function NP = PrepareNetPackage()

% This scripts converts .nnet file of the network into Matlab object.
% To run the script change path to the .nnet file on line 8; 
% set path to the normalization statistics (path to the shared directory) on line 16; 
% and set the path to the shared directory on line 30 so the matlab model is stored there.

    nnetFile = 'E:/VAD_export/STEP2_Torch/models/SigmoidOut/40.nnet';
    NN_package = containers.Map();
    NN_package('FS') = 16000;
    NN_package('window') = hamming(512);
    NN_package('shift') = 128;
    NN_package('nfft') = 512;
    NN_package('lambda') = 0.3;

    load('E:/VAD_export/sharedData/STATS.mat');    
    NN_package('STDin') = 1 ./ STD;
    NN_package('MEANin') = MEAN;    
    
    [isize,noLayers,noNeurons,biases,weights] = loadNNET(nnetFile);
    NN_package('W_in') = cell2mat(weights(1));
    NN_package('B_in') = cell2mat(biases(1));
    NN_package('W_HL1') = cell2mat(weights(2));
    NN_package('B_HL1') = cell2mat(biases(2));
    NN_package('W_HL2') = cell2mat(weights(3));
    NN_package('B_HL2') = cell2mat(biases(3));
    NN_package('W_out') = cell2mat(weights(4));
    NN_package('B_out') = cell2mat(biases(4));

    save('E:/VAD_export/sharedData/sigOut.mat','NN_package');    
end
function [isize,noLayers,noNeurons,biases,weights] = loadNNET(nnetFile)
    fileID = fopen(nnetFile,'r');
    [x] = fread(fileID,2,'uint32');

    if not( strcmp(dec2hex(x(1)),'54454E4E') && (x(2)==0) ) % header verification
        fclose(fileID);
        ans = 'Not a valid NNET file -- see the header' 
    else
        isize = fread(fileID,1,'uint32');   % # of input features
        noLayers = fread(fileID,1,'uint32');% # of layers (in+out+hidden)

        noNeurons = zeros(noLayers,1);      % # of neurons per layer
        for I = 1:noLayers
           noNeurons(I) = fread(fileID,1,'uint32');
        end

        for I = 1:noLayers                  % bias vectors per layer
           biases{I} = fread(fileID,noNeurons(I),'float')';
        end

        sizes = zeros(noLayers,2);          % sizes of weight matrices
        sizes(1,:) = [ isize , noNeurons(1) ];
        for x=1:noLayers-1
            sizes(x+1,:) = [ noNeurons(x) , noNeurons(x+1)];
        end

        for I = 1:noLayers                  % weight matrices per layer
            weights{I} = reshape( fread( fileID , sizes(I,1)*sizes(I,2) , 'float') , [sizes(I,1),sizes(I,2)] );
        end    

        fclose(fileID);
    end
end