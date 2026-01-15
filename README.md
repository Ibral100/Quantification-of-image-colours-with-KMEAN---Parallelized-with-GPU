GPU optimization project for image color quantization using the K-Means algorithm with CUDA and Numba.


📋 Description

This project implements the K-Means algorithm to reduce the number of colors in an image. It offers two versions:

Sequential version (CPU): classic implementation with NumPy
Parallel version (GPU): optimized implementation with CUDA via Numba


Features
 
Image color reduction (quantization)
Custom K-Means clustering
Parallel GPU computation with CUDA

GPU Optimization

The GPU version parallelizes costly operations:

kernel_calcul_distances_affectation: computes distances and assigns each pixel to a cluster in parallel
kernel_calcul_nouveaux_centres: recomputes cluster centers in parallel
