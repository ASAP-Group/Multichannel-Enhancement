
-- LM --- Dataset Utils

-- function computing CMS
function applyCMS(fvec, nSamples) 

  local cms = torch.Tensor(nSamples, settings.inputSize):zero();
  local cmsCounter;
  
  -- indices for computation
  local startIndex, endIndex;
    
  for i = 1, nSamples, 1 do
    -- normal data
    cmsCounter = settings.cmsSize + 1;
    startIndex = i - (settings.cmsSize/2);
    endIndex = i + (settings.cmsSize/2);
    
    -- start
    if(startIndex < 1) then
      cmsCounter = cmsCounter + startIndex - 1;
      startIndex = 1;
    end
    
    -- end
    if(endIndex > nSamples) then
      cmsCounter = cmsCounter - (endIndex - nSamples);
      endIndex = nSamples;
    end
    
    -- compute CMS for given frame   
    cms[{ i, {} }] = (torch.sum(fvec[{ {startIndex, endIndex}, {} }], 1)):div(cmsCounter);
  end
  
  -- apply CMS
  fvec:add(-cms);
  
  return fvec;
  
end  

-- function cloning borders - inputs
function cloneBordersInputs(data, fvec) 
  
  local pre = torch.Tensor(data, 1, torch.LongStorage{1, settings.inputSize});
  local post = torch.Tensor(data, (fvec:size(1) - 1) * settings.inputSize + 1, torch.LongStorage{1, settings.inputSize});   
  
  pre = pre:repeatTensor(settings.seqL, 1);
  post = post:repeatTensor(settings.seqR, 1);   
  
  fvec = torch.cat(pre, torch.cat(fvec, post, 1), 1);
  
  return fvec;
  
end

-- function cloning borders - refs
function cloneBordersRefs(data, fvec) 
  
  local curOut = torch.Tensor(data:size(1) + settings.seqL + settings.seqR);   
  
  for i = 1, settings.seqL, 1 do
    curOut[i] = data[1];
  end  
  for i = settings.seqL + 1, curOut:size(1) - settings.seqR - settings.seqL, 1 do
    curOut[i] = data[i];
  end  
  for i = curOut:size(1) - settings.seqR - settings.seqL + 1, curOut:size(1) - settings.seqL, 1 do
    curOut[i] = data[data:size(1)];
  end    
  
  return curOut;   
  
end

-- function applying framestats inputs -/+ (ln(framestats) / ln(count))
function applyFramestats(inputs, framestats, count, operation)
  
  framestats:log();
  framestats = framestats:repeatTensor(inputs:size(1), 1);
  count = math.log(count);
  
  if (operation == 0) then
    inputs = inputs - (framestats / count);
  elseif (operation == 1) then 
    inputs = inputs + (framestats / count);
  else
    error('Operation: not supported');
  end
  
  return inputs;
  
end
