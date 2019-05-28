-- settings for runFwdPass --
-- settings for runFwdPass --

function Settings(decode)
  local mainPath = "/media/Terezka/VAD_export/STEP2_Torch/";
  local settings = {};
  
  -- general DNN settings
  settings.inputSize = 257;  -- délka feature vektoru, ne šířka vstupní vrstvy
  settings.outputSize = 514;
  
  settings.noEpochs = 40;
  settings.startEpoch = 0;
  settings.batchSize = 1024;
  settings.FWbatchSize = 4096;
  settings.seqL = 0;                  -- number of frames added as input - left
  settings.seqR = 0;                  -- number of frames added as input - right
  settings.learningRate = 0.01;
  
  settings.lrDecayActive = 0;
  if(settings.lrDecayActive == 1) then
    settings.lrDecay = 0.1;
  end  
  
  
  
  settings.listsTrain   = {'train.flst'};
  settings.listsValTest = {'valid.flst','test.flst'};
  
  
  settings.dropout = 0;             
  settings.computeStats = 0; 
  settings.model = "special_sig"
  
  -- other settings
  settings.cuda = 1;
  settings.shuffle = 1;
  settings.exportNNET = 1;
  settings.inputType = "JM_specGram";        
  settings.refType = "JM_BinLabel"       
  
  
  settings.spcGrmExt = ".nin";
  settings.refExt = ".05o";
  settings.listFolder = mainPath; 

  settings.modelName = "sigmOut"
  settings.outputFolder = mainPath .. "models/";
  
  settings.logFolder = "/log/";
  settings.modFolder = "/sigmoidOut/";
  settings.myLogFolder = "/myLogs/";
  settings.logPath = "settings.log";
  
  settings.scriptFolder = mainPath .. "scripts/";  

  -- log  
  flog = logroll.file_logger(settings.outputFolder .. settings.logFolder .. settings.logPath);
  flog.info(settings);

  -- create output folder 
  os.execute("mkdir -p " .. settings.outputFolder .. settings.modFolder);
  
  return settings;
    
end
