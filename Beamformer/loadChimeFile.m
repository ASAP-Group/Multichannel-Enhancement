function [s] = loadChimeFile(speechCH1Path)

channelNum = 6;

[tmps] = audioread(speechCH1Path);

s = zeros(size(tmps,1),channelNum);

s(:,1) = tmps;
for lpCh = 2:channelNum
    
    currPath = strrep(speechCH1Path,'CH1',['CH' num2str(lpCh)]);
    s(:,lpCh) = audioread(currPath);

end    