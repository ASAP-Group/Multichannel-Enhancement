clear *; 
close all;
clc;

% File given in the variable "inputFile" on line 7 is path to 1st channel 
% of CHiME audio recording.
% Function loadChimeFile loads all the channels of the recording and puts 
% them in the "x" variable.
% When processing another input data, this function is to be replaced.
% "x" has to be a matrix where each column represents recording from one
% microphone.

inputFile = '..\mVAD\sharedData\test\F05_440C0207_CAF.CH1.wav';
    
[x] = loadChimeFile(inputFile);

BF = 'FSB';
VADsel = 'mVAD';

settings.BMstructure = 'onereference';
settings.postfilter = 'post2';
settings.vnad_median = true;
settings.thrs = 0.05;
settings.blockn = 100;
settings.nsubblock = 10;
settings.tchan = 4;

tic
[s,ysc]=chimeenhancer( x(:,[1 3 4 5 6]) , BF , VADsel , settings );
toc

soundsc(s,16000)