"""Drone RF detection and classification: spectrogram (2D) and IQ (1D) CNNs.

Re-implementation of the pipelines from
  Gluege et al., "Robust Low-Cost Drone Detection and Classification Using
  Convolutional Neural Networks in Low SNR Environments", IEEE JRFID 2024
  (spectrogram VGG models, dataset v2), and
  Gluege et al., "Robust Drone Detection and Classification from Radio
  Frequency Signals Using Convolutional Neural Networks", NCTA 2023
  (IQ 1D VGG vs spectrogram 2D VGG, dataset v1).
"""
