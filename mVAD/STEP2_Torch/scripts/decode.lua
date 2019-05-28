
-- LM --- Compute outputs

-- general libraries
require 'torch'
require 'paths'
require 'xlua'
require 'math'
require 'logroll'
require 'nn'

-- require settings
if (arg[1]) then 
  assert(require(string.gsub(arg[1], ".lua", "")));
else
  require 'settings'
end

-- initialize settings
settings = Settings(true);    

-- add path to scripts
if (settings.scriptFolder) then  
  package.path = package.path .. ";" .. settings.scriptFolder .. "?.lua";
end

-- program requires
require 'dataset'
require 'utils'
require 'io-utils'
require 'set-utils'
require 'nn-utils'

-- load modules for cuda if selected
if (settings.cuda == 1) then
  require 'cunn'
  require 'cutorch'  
end

-- load model from file
if not paths.filep(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod") then  
  error('File ' .. settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod" .. ' does not exist!');
end
model = torch.load(settings.outputFolder .. settings.modFolder .. settings.startEpoch .. ".mod");

-- load stats
if (settings.computeStats == 1) then
  settings.mean = readStat(settings.outputFolder .. settings.statsFolder .. '/mean.list');
  settings.var = readStat(settings.outputFolder .. settings.statsFolder .. '/std.list')
end

-- load framestats
if (settings.applyFramestats == 1) then
  framestats, count = loadFramestats(settings.outputFolder .. settings.statsFolder .. '/framestats.list');
end

-- cuda on/off
if (settings.cuda == 1) then 
  model:cuda();
  modelC = nn.Sequential();
  modelC:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));
  modelC:add(model);
  modelC:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'));     
else
  modelC = mlp;   
end

-- initialize logs
flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. '/decode.log');
plog = logroll.print_logger();
log = logroll.combine(flog, plog);

-- evaluation mode
modelC:evaluate()

if not paths.filep(settings.listFolder .. settings.decodeFile) then  
  error('File ' .. settings.listFolder .. settings.decodeFile .. ' does not exist!');
end

-- decode per file
for line in io.lines(settings.listFolder .. settings.decodeFile) do

  -- prepare datasets
  dataset = Dataset(line, 0, 1, 0); 
  log.info("Decoding file: " .. line .. " type: " .. settings.decodeType);
  
  -- compute number of batches
  noBatches = (dataset:size() - dataset:size() % settings.batchSize) / settings.batchSize;
  
  -- prepare output file
  folders = split(line, "/");
  fileParts = split(folders[#folders], ".");
  os.execute("mkdir -p " .. settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1]);
  
  -- according to requested decode type
  if (settings.decodeType == "lkl") then
    ff = torch.DiskFile(settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1] .. "/" .. fileParts[1] .. "." .. settings.decodeType, "w");
    ff:binary();
    saveHTKHeader(ff, dataset:size());
  elseif (settings.decodeType == "txt") then
    ff = io.open(settings.outputFolder .. settings.decodeFolder .. "/" .. folders[#folders-1] .. "/" .. fileParts[1] .. "." .. settings.decodeType, "w");
  else
    error('DecodeType: not supported');
  end
  
  -- process batches
  for noBatch = 1, noBatches + 1, 1 do
    
    -- last batch fix
    batchSize = settings.batchSize;
    if (noBatch == noBatches + 1) then
      batchSize = dataset:size() - (noBatches * settings.batchSize);
    end
    
    -- prepare input tensor
    local inputs = torch.Tensor(batchSize, settings.inputSize * (settings.seqL + settings.seqR + 1)):zero();
    
    -- process batches
    for i = 1, batchSize, 1 do
      local index = (noBatch - 1) * batchSize + i;     
      ret = dataset:get(index);
      inputs[i] = ret.inp;
    end
    
    -- feed forward
    local pred = modelC:forward(inputs);  
    
    -- normalize using framestats
    if (settings.applyFramestats == 1) then
      pred = applyFramestats(pred, framestats, count, settings.applyFramestatsType);
    end
    
    -- save output
    if (settings.decodeType == "lkl") then
      ff:writeFloat(pred:storage());
    elseif (settings.decodeType == "txt") then
      for i = 1, batchSize, 1 do
        _, mx = pred[i]:max(1);
        ff:write(mx[1]-1 .. "\n");
      end
    end  
      
  end
  
  ff:close();

end
