
-- LM --- Datasets

function Dataset(fname, isFileList, decode, computeFramestats)
  
  -- initialization
  local dataset = {}
  
  -- logs
  if (decode == 0) then
    flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/' .. fname .. '.log');
    plog = logroll.print_logger();
    log = logroll.combine(flog, plog);
  end
    
  -- set tensors to float to handle data
  torch.setdefaulttensortype('torch.FloatTensor');
  
  -- log & timer
  if (decode == 0) then
    log.info('Preparing dataset: ' .. fname);
  end
  local begin = sys.clock();
  
  -- dataset initialization
  dataset.index = {};
  dataset.nSamplesList = {};
  dataset.cache = {};
  dataset.nSamples = 0;
  
  -- initialize sample count
  local totalSamples = 0;
  
  -- load filelist
  local fileList = {};
  if (isFileList == 1) then
    fileList = readFileList(settings.listFolder .. fname);
  else
    table.insert(fileList, fname);
  end
  

  -- read data from files
  for file = 1, #fileList, 1 do  
    
    -- log
    if (decode == 0) then
      flog.info('Processing file: ' .. file);
    end
        
    -- read input files
    local nSamples, sampPeriod, sampSize, parmKind, data, fvec;
    if (settings.inputType == "htk") then	  
      nSamples, sampPeriod, sampSize, parmKind, data, fvec = readHTK(fileList[file]);	  
    elseif ( settings.inputType == "MB_specGram" ) then
      nSamples, fvec = readMB_specGram(fileList[file])
    elseif ( settings.inputType == "JM_specGram" ) then
      nSamples, fvec = readJM_specGram(fileList[file])
    else
      error('InputType: not supported');
    end
        
    -- fix for DNN alignment by ntx4
    if (settings.dnnAlign == 1) then
      nSamples = nSamples - 1;
    end

    -- read ref outputs
    local currentOutput;
    if (decode == 0) then 
      if (settings.sameFolder == 0) then
        fileList[file] = fileList[file]:gsub(settings.parPath, settings.refPath);
      end
      
      if (settings.refType == "akulab") then
        currentOutput = readAkulab(fileList[file], nSamples);
      elseif (settings.refType == "rec-mapped") then
        currentOutput = readRecMapped(fileList[file], nSamples);
      elseif (settings.refType == "MB_BinLabel") then
        currentOutput,nSamples = readFBinMask(fileList[file]);		
      elseif (settings.refType == "JM_BinLabel") then
        currentOutput,nSamples = read_JM_FBinMask(fileList[file]);	        
      else
        error('RefType: not supported');
      end
      
 
      -- sanity check - input/ref sample size + ntx4 fix
      if (currentOutput:size(1) == nSamples +1) then
        currentOutput = currentOutput[{{1, nSamples}}];
      elseif (currentOutput:size(1) ~= nSamples) then
        error('Nonmatching sample count' .. fileList[file]);
      end
    end
  
    -- save CMS processed data to cache table
    fvec = fvec:view(fvec:size(1) * settings.inputSize);
    table.insert(dataset.cache, {inp = fvec, out = currentOutput});
    
    -- save counts of samples per file to table
    nSamples = nSamples - settings.seqL - settings.seqR;
    table.insert(dataset.nSamplesList, nSamples);
    
    -- calculate total samples
    if (nSamples >= 1) then
      totalSamples = totalSamples + nSamples;
    end
    
  end
    
  -- prepare tensors for data for training
  dataset.index.file = torch.Tensor(totalSamples):int();    -- tensor -> frames vs. file
  dataset.index.pos = torch.Tensor(totalSamples):int();     -- tensor -> frames vs. position in file
  
  -- fill tensors accordingly to allow training
  for ll = 1, #dataset.nSamplesList, 1 do   
    local nSamples = dataset.nSamplesList[ll];
    if (nSamples >= 1) then     -- sanity check
      i = settings.seqL;
      dataset.index.file:narrow(1, dataset.nSamples + 1, nSamples):fill(ll);              -- fe [1 1 1 2 2 2 2 2]
      dataset.index.pos:narrow(1, dataset.nSamples + 1, nSamples):apply(function(x)       -- fe [1 2 3 1 2 3 4 5] + seqL
        i = i + 1;
        return i;
      end);
      dataset.nSamples = dataset.nSamples + nSamples;   -- compute final number of samples
    end
  end  
  
  
  
  if settings.computeStats == 1 then
    -- prepare mean & std tensors
    local mean = settings.mean:repeatTensor(settings.seqL + settings.seqR + 1);
    local var = settings.var:repeatTensor(settings.seqL + settings.seqR + 1);
  end
  
  
  
  -- log time
  --if (decode == 0) then
  --  log.info('Dataset prepared in ' .. sys.clock() - begin);
  --end
  
  -- return number of samples
  function dataset:size() 
    return dataset.nSamples;
  end
  
  -- return frame + surroundings and akulab
  function dataset:get(i)
    
    -- identify file
    local fileid = dataset.index.file[i];
    
    -- load file data
    local currentInput = self.cache[fileid].inp;
    local currentOutput = self.cache[fileid].out;
    
    -- find the indices of asked data
    local startIndex = (dataset.index.pos[i] - settings.seqL - 1) * settings.inputSize + 1;
    local endIndex = startIndex + (settings.seqR + 1 + settings.seqL) * settings.inputSize - 1;
    

    
    
    -- clone the asked data   
    local inp = currentInput[{{startIndex, endIndex}}]:clone();


  if settings.computeStats == 1 then
    -- normalize
    inp:add(-mean);
    inp:cdiv(var);  
  end
    
    
    -- prepare the output
    local out;
    if (decode == 0) then
      --out = (currentOutput[dataset.index.pos[i]] + 1); *** tady ***
      out = (currentOutput[dataset.index.pos[i]]);
    end
    
    -- return the asked data
    return {inp = inp, out = out};
    
  end
  
  return dataset;
  
end
