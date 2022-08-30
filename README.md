# ARWaveletCNN
Adversarialy Robust WaveletCNN Code for our paper under review.

The architecture of our ARWaveletCNN model is as follows:
![](ARWaveletCNN.png)

This repository contains the code for our proposed ARWaveletCNN model, as well as DenseNet121 best performing model and their respective evaluations. Code also contains code to reproduce all of our experiments in the paper.
For the sake of analysis and to fully understand how the ARWaveletCNN model works, we include the following visualizations:
  - A visualization of the Harmonic and percussive parts of several different audio files (including augmented audio using https://github.com/fathana/spoofAugment)
  - A visualization of several wavelet transforms (e.g. Haar, db2, dmey, etc.) comprising the different channels
  - A comprehensive visualization of the generated feature maps/activations of ARWaveletCNN (including all successive layers) using several different input files (bonafide, spoof, augmented, correctly classified, and misclassified samples) which clearly shows that the ARWaveletCNN architecture works by eliminating all seemingly normal features throughout all the successive layers, and keeping all appearently anomalous features in the original input. In this case, a final blank output feature map (last layer) indicates that the sample is bonafide, otherwise a presence of residue of anomalies/impurities in the final output indicates that the input has been compromised (spoofed).
