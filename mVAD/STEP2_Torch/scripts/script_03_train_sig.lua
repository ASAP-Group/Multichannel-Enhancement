-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'nn'
require 'logroll'
require 'gnuplot'
require 'os'
require 'io-utils'

torch.setdefaulttensortype('torch.FloatTensor');

-- require settings
if (arg[1]) then 
  assert(require(string.gsub(arg[1], ".lua", "")));
else
  local settings = require 'settings_sigOut';
end

-- initialize settings
settings = Settings();   
-- add path to scripts
if (settings.scriptFolder) then  
  package.path = package.path .. ";" .. settings.scriptFolder .. "?.lua";
end

  
-- program requires
require 'dataset'
require 'utils'
require 'nn-utils'

-- load modules for cuda if selected
if (settings.cuda == 1) then
  require 'cunn'
  require 'cutorch'  
end

-- Initializations
    -- initialize logs
    flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/train.log');
    plog = logroll.print_logger();
    log = logroll.combine(flog, plog);
    
    -- initialize the network
    if (settings.startEpoch == 0) then
      if (settings.model == "classic") then
        model = buildFFModel();
      elseif (settings.model == "residual") then
        model = buildResidualModel();
      elseif (settings.model == "special_sig") then
        model = build_Spec_sig();
      else
        error('Model: not supported');
      end
    -- load epoch to continue training 
    else
      if paths.filep(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod") then  
        model = torch.load(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod");
        log.info('Epoch ' .. settings.startEpoch .. ' loaded');
      else
        log.error('Epoch ' .. settings.startEpoch .. ".mod" .. ' does not exist!');
      end
    end

  
  criterion = nn.MSECriterion();

    -- cuda on/off
    if (settings.cuda == 1) then
      
      criterion:cuda();      
      modelC = model:cuda();      
           
      flog.info("Using CUDA"); 
    else
      modelC = model;   
      flog.info("Using CPU");
    end
    
        



-- loading of dataSets
  trainDataSet = Dataset(settings.listsTrain[1], 1, 0, 0);
  local noTrainBatches = (trainDataSet:size() - trainDataSet:size() % settings.batchSize) / settings.batchSize;
  testSets = {};  
  for I = 1, #settings.listsValTest, 1 do
    testSets[I] = (Dataset(settings.listsValTest[I], 1, 0));
  end

-- FOR cycle through epochs
for epoch = settings.startEpoch + 1, settings.noEpochs, 1 do
    shuffle = torch.randperm(trainDataSet:size(), 'torch.LongTensor'); -- shuffle training data 
    modelC:training() -- mode training
    log.info("Training epoch: " .. epoch); 
  
      -- training per batches
      for noBatch = 1, noTrainBatches, 1 do

        -- prepare inputs & outputs tensors    
        local inputs = torch.Tensor(settings.batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
        local targets = torch.Tensor(settings.batchSize, settings.outputSize):zero();
  
        -- process batches
        for i = 1, settings.batchSize, 1 do    
          -- pick frame 
          local index = (noBatch - 1) * settings.batchSize + i;     
          index = shuffle[index];

          ret = trainDataSet:get(index); -- retrieve data for selected frame, fill input arrays for training
          
          inputs[i] = ret.inp;
          targets[i] = ret.out;
        end 
        
        if settings.cuda == 1 then
          inputs = inputs:cuda()
          targets = targets:cuda()
        end

        -- forward propagation  
        criterion:forward(modelC:forward(inputs), targets);        
        
        modelC:zeroGradParameters(); -- zero the accumulation of the gradients
        modelC:backward(inputs, criterion:backward(modelC.output, targets)); -- back propagation
        
        -- update parameters
        if (settings.lrDecayActive == 1) then 
          learningRate = settings.learningRate / (1 + (epoch - 1) * settings.lrDecay); 
          modelC:updateParameters(learningRate);
        else 
          modelC:updateParameters(settings.learningRate);  
        end
        
      end         
  
    -- logs & export model
    plog.info("Saving epoch: " .. epoch .. "/" .. settings.noEpochs);
    torch.save(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", modelC);
    if (settings.exportNNET == 1) then
      exportModel(settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".mod", settings.outputFolder .. settings.modFolder .. "/" .. epoch .. ".nnet");
    end
  


    -- evaluation    
    if (#settings.listsValTest > 0) then
      plog.info("Testing epoch: " .. epoch .. "/" .. settings.noEpochs);       
      modelC:evaluate(); -- mode evaluation 
      
      for I = 1, #settings.listsValTest, 1 do
        sets = testSets[I];
        local noBatches = ((sets:size() - sets:size() % settings.FWbatchSize) / settings.FWbatchSize);          
               
        -- reset temp data
        local RMSEr = 0.0;
        local count = 0.0;
                
        -- prepare inputs & outputs tensors 
        local inputs = torch.Tensor(settings.FWbatchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
        local targets = torch.Tensor(settings.FWbatchSize, settings.outputSize):zero();
    
        for j = 1, noBatches, 1 do              
          -- process batches
          for k = 1, settings.FWbatchSize, 1 do
            local index = (j - 1) * settings.FWbatchSize + k; -- pick frame and obtain data
            ret = sets:get(index);
            inputs[k] = ret.inp; targets[k] = ret.out;
          end
          
          if settings.cuda == 1 then
            inputs = inputs:cuda()            
          end          
          
          local outputs = modelC:forward(inputs); -- forward pass
          
          if settings.cuda == 1 then
            outputs = outputs:typeAs(targets)
          end    

          local diff = outputs:csub(targets);
          local rmse = torch.pow(diff,2.0);
          
          RMSEr = RMSEr + rmse:sum();
          count = count + targets:nElement() * settings.FWbatchSize;

        end        
        
        -- print and dump result
        print ('RMSError ' .. settings.listsValTest[I] .. ' :: ' .. RMSEr/count);         
        
      end
      
    end  
end


