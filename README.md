# Novel View Synthesis with NeRF

In the task of novel view synthesis, your training set consists of a set of images of a scene where you know the camera parameters (intrinsic and extrinsic) for each image. Your goal is to build a model that can synthesize images showing the scene from new viewpoints unseen in the training set.

Over the past two years, Neural Radiance Fields (NeRFs) have emerged as a simple and powerful model for this problem. NeRFs rely on the idea of volume rendering: to determine the color of a pixel, we shoot a ray originating from the camera center through the pixel and into the scene; for a set of points along the ray, we compute both the color and opacity of the 3D scene. Integrating the influence of these points gives the color of the pixel. The original NeRF paper [Mildenhall et al, ECCV 2020] proposed to train a fully-connected neural network that inputs (x, y, z) and a viewing direction, and outputs the RGB color and opacity of the 3D scene at that point. This network is trained to reproduce the pixel values of the training images; during inference, the network can be used to synthesize the color of pixels in novel views unseen during training.

The goal of this project is to implement a NeRF model and reproduce some of the main results from the original NeRF paper. You may optionally also incorporate ideas from some followup papers.

Note that this project is likely to involve a more complex implementation than the other two suggested projects. It is also intended to be somewhat open-ended; while the goal is to re-implement NeRF, you can be creative in exactly what results you show, and how you deviate from the original NeRF paper.

## Papers to read
Mildenhall et al, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis”, ECCV 2020
Barron et al, “Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields”, ICCV 2021
