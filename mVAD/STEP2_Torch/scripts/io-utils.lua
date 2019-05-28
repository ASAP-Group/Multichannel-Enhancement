function readJM_specGram(file)

  local fff = (file .. settings.spcGrmExt) 
  
  --print (fff)
  
  if not paths.filep(fff) then  
    error('File ' .. fff .. ' does not exist!');
  end 
  
  local f = torch.DiskFile(file .. settings.spcGrmExt, 'r');
  f:binary();
    
  local nSamples = f:readInt(); 
  local data = f:readFloat(nSamples);
  f:close();

  local fvec = torch.Tensor( data , 1, torch.LongStorage{nSamples/257 , 257} );

  return fvec:size(1), fvec
  
end

function read_JM_FBinMask(file)
  
  local fff = (file .. settings.refExt)
  --print (fff)

  if not paths.filep( fff ) then  
    error('File ' .. fff .. ' does not exist!');
  end

  local f = torch.DiskFile(file .. settings.refExt, 'r');
  f:binary();
  
  local nSamples = f:readInt();
  
  local data = f:readFloat(nSamples); 
  f:close();
  
  local fvec = torch.Tensor( data , 1, torch.LongStorage{nSamples/514 , 514} );
  
return fvec,fvec:size(1)
  
end

-- function loading filelist
function readFileList(fileList)
  
  torch.setdefaulttensortype('torch.FloatTensor');

  if not paths.filep(fileList) then  
    error('File ' .. fileList .. ' does not exist!');
  end

  local files = {};
  for line in io.lines(fileList) do
    table.insert(files, line);
  end
  
  return files;
  
end

-- function exporting nnet file
function exportModel(ifile, ofile)
  
  local mlp = torch.load(ifile);
  local linearNodes = mlp:findModules('nn.Linear');
  local f = torch.DiskFile(ofile, "w");
  f:binary();
  f:writeInt(0x54454E4e);
  f:writeInt(0);
  local isize = linearNodes[1].weight:size(2);
  f:writeInt(isize);
  local noLayers = #linearNodes;
  f:writeInt(noLayers); 
  for i = 1, noLayers do
    -- m, n = linearNodes[i].weight:size();
    local noNeurons = linearNodes[i].weight:size(1);
    f:writeInt(noNeurons);
  end
  for i = 1, noLayers do
    local stor = linearNodes[i].bias:float():storage();
    f:writeFloat(stor);
  end      
  for i = 1, noLayers do
    local stor = linearNodes[i].weight:float():storage();
    f:writeFloat(stor);
  end
  f:close();
  
end

-- function to save stats
function saveStat(file, stat)
  
  local f = torch.DiskFile(file, 'w');
  
  for v = 1, stat:size(1), 1 do
    f:writeFloat(stat[v]);
  end
  
  f:close()
  
end

-- function loading stats
function readStat(file)

  torch.setdefaulttensortype('torch.FloatTensor');
  
  local f = torch.DiskFile(file, 'r');
  local stat = f:readFloat(settings.inputSize);
  stat = torch.Tensor(stat, 1, torch.LongStorage{settings.inputSize});
  
  torch.setdefaulttensortype('torch.DoubleTensor');
  
  return stat;
  
end