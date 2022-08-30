# ARWaveletCNN
Adversarialy Robust WaveletCNN Code for our paper under review.

The architecture of our ARWaveletCNN model is as follows:
![](ARWaveletCNN.png)

This repository contains the code for our proposed ARWaveletCNN model, as well as DenseNet121 best performing model and their respective evaluations. Code also contains code to reproduce all of our experiments in the paper.
For the sake of analysis and to fully understand how the ARWaveletCNN model works, we include the following visualizations:
  - A visualization of the Harmonic and percussive parts of several different audio files (including augmented audio samples using https://github.com/fathana/spoofAugment)
  - A visualization of several wavelet transforms (e.g. Haar, db2, dmey, etc.) comprising the different channels
  - A comprehensive visualization of the generated feature maps/activations of ARWaveletCNN from input to output (including all successive layers) using several different input files (bonafide, spoof, augmented, correctly classified, and misclassified samples) which clearly shows that the ARWaveletCNN architecture works by eliminating all seemingly normal features throughout the successive layers, and keeping all appearently anomalous distortions/features (= indicators of a flawed fake audio) in the original input. In this case, a final blank output feature map (absence of anomalies in the last layer) indicates that the audio sample is bonafide (perfect symmetry between the successive wavelet transforms and the convolutional filters), otherwise a presence of residue of anomalies/impurities in the final feature map indicates that the input has been altered (spoofed).
