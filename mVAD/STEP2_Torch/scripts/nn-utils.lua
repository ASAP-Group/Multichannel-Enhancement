-- network initialization - literature
function initializeLL(inputSize, outputSize)
  
  local l = nn.Linear(inputSize, outputSize)
  local v = math.sqrt(6.0 / (outputSize + inputSize));
  l.weight = torch.randn(outputSize, inputSize);
  l.weight:mul(v);
  l.bias:zero();
  return l;
    
end

function build_Spec_sig()
  -- input layer
  model = nn.Sequential();
  model:add(initializeLL( 257 , 1024 ));   -- layer size
  model:add(nn.ReLU());   -- layer type
  
  -- hidden L1
    model:add(initializeLL( 1024 , 1024 ));    -- layer size
    model:add(nn.ReLU());   -- layer type    
  
  -- hidden L2
    model:add(initializeLL( 1024 , 1024 ));    -- layer size
    model:add(nn.ReLU());   -- layer type    
    
  -- output layer
  ll = nn.Linear( 1024 , 514 );   -- layer size
  ll.weight:zero();   -- default weights
  ll.bias:zero();     -- default bias
  model:add(ll);
  model:add(nn.Sigmoid());
  
  return model;    
  
end