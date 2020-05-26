# Block-Online Multi-Channel Speech Enhancement Using DNN-Supported Relative Transfer Function Estimates

This project is an implementation of the multi-channel enhancement technique described in the paper:

Jiri Malek, Zbynek Koldovsky and Marek Bohac, “Block-Online Multi-Channel Speech Enhancement Using DNN-Supported Relative Transfer Function Estimates”, IET Signal Processing, vol. 14, no. 3, pp. 124–133, May 2020.

The paper is available on arXiv:

http://arxiv.org/abs/1905.03632

Please cite the above stated reference, if you use the beamformer in your scientific research.

The technique is intended for enhancement of multi-channel speech recordings with single active speaker and real-world background noise.

## Getting Started

To run the enhancer, the Matlab environment is needed. 
An example of enhancement setup is provided by script "\Beamformer\run_Beamformer.m".

Functional Voice Activity Detector (VAD) is provided in directory "\Beamformer\VADresources\".

To retrain the VAD, Torch 7 environment is required (and possibly CUDA for GPU computation).
The datails of the training procedure are given in the file "\mVAD\CookBook.txt" .
The training scripts are designed to utilize the CHiME-4 challenge training speech recordings.

## Authors

Jiri Malek

Zbynek Koldovsky

Marek Bohac

## License

The enhancer is free to use for non-commertial and scientific puproses and is licensed to Technical University in Liberec.
Details of the license are provided in the file \license.txt .

The VAD training is intended for utilization on the CHiME-4 challenge training data.
It uses some of the scripts provided by the challenge organizers and therefore is subject to GNU3 license as CHiME-4 data.
Details of the licence are provided in the file \mVAD\STEP1_dataPreparation\utils\_license_statement_.txt .
