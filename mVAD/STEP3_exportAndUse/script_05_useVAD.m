clc; clear all; close all

% This script shows, how to call the VAD network and isolate VAD and NAD outputs.
%  Variable "recording" specifies whole path to the processed audio file.
% Path given on line 10 leads to saved matlab model of the VAD network.

recording = '..\sharedData\test\F05_440C0207_CAF.CH1.wav';
[x,~] = audioread(recording);

NN_pack = load('E:\VAD_export\sharedData\sigOut.mat');
NN_pack = NN_pack.NN_package;

netOut = runVAD( x , NN_pack );

VAD = netOut(1:257,:);
NAD = netOut(258:end,:);

imshow(mat2gray(netOut))