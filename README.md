# LOW-LIGHT IMAGE ENHANCEMENT

### UPC School
### Authors: Alberto Camacho, Marcos Carlevaro, Vanessa Castillo, Pablo López
### Advisor: Dani Fojo

---

## Index

- Introduction
- Datasets
- About the model
- Model Training
- Results
- Conclusions
- Execution Instructions
- App Usage

## Introduction

In this project we implement RetinexNet with PyTorch based on this paper:

[Deep Retinex Decomposition for Low-Light Enhancement](https://paperswithcode.com/paper/deep-retinex-decomposition-for-low-light)

We know that insufficient lighting can significantly degrade the visibility of images and that there are many reconstruction/enhancing techniques in order to get better quality.  

<img src="figs/low-normal-light-images.png"/>  

We chose Retinex model as an effective tool for low-light image enhancement. One of the main reasons was that it assumes that observed images can be decomposed into their reflectance and luminance, a theory that was postulated by Edwin Herbert Land which denies trichromatic Newton color theory and states that color is a brain active composition done by comparison.  

<img src="figs/color_theory.png"/>  

[¿Qué es el color? La teoría Retinex de Land](http://opticaluzycolor.blogspot.com/2011/03/que-es-el-color-la-teoria-retinex-de.html?m=1)

<img src="figs/reflectance-luminance.PNG"/>

## Datasets

We used both datasets already used in the original paper:

Dataset 1: [LOL (LOw-Light dataset)](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)  
500 real images (485 to train and 15 to test), 400x600px.

<img src="figs/lol-dataset.PNG"/>

Dataset 2: [Synthetic pairs](https://drive.google.com/file/d/1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F/view)  
1250 images (1000 to train and 250 to test), different sizes.

<img src="figs/synthetic-dataset.PNG"/>

And we have created our own set of 10 pair of images with normal and low light to test the final model, these are some examples:

<img src="figs/readme_example.png" width="200px"/>

## About the model
In the figure below we can find the proposed framework for Retinex-Net. The enhancement process is divided into three steps: decomposition, adjustment and reconstruction. In the decomposition step, a subnetwork Decom-Net decomposes the input image into reflectance and illumination. In the following adjustment step, an encoder-decoder based Enhance-Net brightens up the illumination. Multi-scale concatenation is introduced to adjust the illumination from multi-scale perspectives. Noise on the reflectance is also removed at this step. Finally, we reconstruct the adjusted illumination and reflectance to get the enhanced result.

<img src="figs/retinexnet.png" width="600px"/>

As illustrated in the figure above, Decom-Net takes the low-light image *S_low* and the normal-light one *S_normal* as input, then estimates the reflectance *R_low* and the illumination *I_low* for *S_low*,
as well as *R_normal* and *I_normal* for *S_normal*, respectively. It first uses a 3×3 convolutional layer to extract features from the input image. Then, several 3×3 convolutional layers with ReLU as the activation function are followed to map the RGB image into reflectance and illumination. A 3×3 convolutional layer projects *R* and *I* from feature space, and sigmoid function is used to constrain both *R* and *I* in the range of *[0, 1]*.  

The loss *L* consists of three terms: reconstruction loss *L_recon*, invariable reflectance loss *L_ir*, and illumination smoothness loss *L_is*:  

<img src="figs/loss.png"/>

where *λ_ir* and *λ_is* denote the coefficients to balance the consistency of reflectance and the smoothness of illumination. 

Based on the assumption that both *R_low* and *R_high* can reconstruct the image with the corresponding illumination map, the reconstruction loss *L_recon* is formulated as:  

<img src="figs/loss_recon.png"/>

Invariable reflectance loss *L_ir* is introduced to constrain the consistency of reflectance:

<img src="figs/loss_ir.png"/>

Illumination smoothness loss Lis is formulated as:

<img src="figs/loss_is.png"/>

where ∇ denotes the gradient, including *∇_h* (horizontal) and *∇_v* (vertical), and *λ_g* denotes the coefficient balancing the strength of structure-awareness.

## Model training

We have trained the net in mainly 4 different ways:

* **Decom-Net and Enhance-Net separately**  
This is how the model is trained in the original paper. In this case there are 2 different optimizers, one for each net. First, the Decom-Net is trained for all the epochs and once the training has finished it is used in eval mode so the Enhance-Net is trained for all the epochs using the Decom-Net already trained. The original model in the paper uses convolutional layers and an increasing resize function using interpolation in order to apply skip connections in the Enhance-Net. In the critical review of the project we decided to change those layers removing the resizing and using transposed convolutional layers intead, so we have the following two experiments:  
    * *With convolutional + resize layers in Enhance-Net*
    * *With convolutional transposed layers in Enhance-Net*
* **Decom-Net and Enhance-Net together**  
We decided to try training both nets at the same time, joining both optimizers and loss functions in order to compare results with the original model. In this case we observed that the results of the Enhance-Net were white images, so the net wasn't contributing much to the final result. We then thought that since the Decom-Net was already obtaining good results with the reflectance image, we could try ignoring the Enhance-Net. Hence we get the following two other experiments:
    * *Training both nets*
    * *Ignoring Enhance-Net*

In the end we haven't used any denoising operation. Both the one referenced in the paper (BM3D) and the alternative we found in the OpenCV library (fastNlMeansDenoising) were complex to apply and didn't add visible improvement in the resulting enhanced images. We haven't done any hyperparameter tunning neither, as we indeed had a loss definition for the model training but no other metric that allowed us to compare which final enhanced images were better, except for our subjective human eye.

In all four experiments we have trained 200 epochs and started with learning rate at 0.001 for both nets, using a scheduler. We tried both StepLR and ReduceOnPlateau, but StepLR delivered better results.

We have used [Wandb](https://wandb.ai/site) in order to check the performance of all the models form the different experiments.

We have trained in [Google Cloud](https://cloud.google.com/?hl=es) with the following properties:

<img src="figs/google-vm-properties.PNG"/>

## Results
For each experiment there is a link to the Wandb report, so we can check the Decom-Net and Enhance-Net output images for every epoch of the training.

### Decom-Net and Enhance-Net separately
* [With convolutional + resize layers in Enhance-Net](https://wandb.ai/aidl_retinex/retinex/reports/Experiment-3--VmlldzoxODYzNDIx?accessToken=60nx4c5yxf7zm8z5cerreejs0hs40oxte0d7kxa0h51x4x34ozmzxm3jdedzxygz)  

<img src="figs/1_losses.png"/>

| | Decom-Net | Enhance-Net |
| ------------    | -----------  | ----------- |
| Training Loss   | 0.008 | 0.120 |
| Validation Loss | 0.008 | 0.134 |

<img src="figs/1_training_decom.png"/>
<img src="figs/1_val_decom.png"/>
<img src="figs/1_training_relight.png"/>
<img src="figs/1_val_relight.png"/>

* [With convolutional transposed layers in Enhance-Net](https://wandb.ai/aidl_retinex/retinex/reports/Experiment-4--VmlldzoxODYzNDQ5?accessToken=6jhwumkcsi5o3ocjh82xc0rdg73v69l39aj8daizksn8zssk6sqlxgucw85rv0k1)  

<img src="figs/2_losses.png"/>

| | Decom-Net | Enhance-Net |
| ------------    | -----------  | ----------- |
| Training Loss   | 0.007 | 0.123 |
| Validation Loss | 0.007 | 0.129 |

<img src="figs/2_training_decom.png"/>
<img src="figs/2_val_decom.png"/>
<img src="figs/2_training_relight.png"/>
<img src="figs/2_val_relight.png"/>

For these 2 exepriments we could see that the results looked really similar, but then comparing some enhanced images from the test set, we could observe that using interpolation instead of transposed convolutional layers had some issues with plain color zones, creating some noise:

<img src="figs/interpolation_problem.png"/>


### Decom-Net and Enhance-Net together
* [Training both nets](https://wandb.ai/aidl_retinex/retinex/reports/Experiment-1--VmlldzoxODYzMzU0?accessToken=uprm4x1mdqf8v52niitxwzlxxokmc2bfglkedinqjpk0887ms3gxqku7wdqp3d56)

<img src="figs/3_losses.png"/>

| | Decom-Net | Enhance-Net |
| ------------    | -----------  | ----------- |
| Training Loss   | 0.014 | 0.096 |
| Validation Loss | 0.014 | 0.107 |

<img src="figs/3_training_phase.png"/>
<img src="figs/3_val_phase.png"/>

* [Ignoring Enhance-Net](https://wandb.ai/aidl_retinex/retinex/reports/Experiment-2--VmlldzoxODYzNDAy?accessToken=fcacjitj4dpgzwmbd31tjfrjckvbpav97me54rk3xirvxznb3c0y3qk677zulto1)

<img src="figs/4_losses.png"/>

| | Decom-Net | Enhance-Net |
| ------------    | -----------  | ----------- |
| Training Loss   | 0.016 | 0.099 |
| Validation Loss | 0.014 | 0.113 |

<img src="figs/4_training_phase.png"/>
<img src="figs/4_val_phase.png"/>  

In the case of training both Decom-NEt and Enhance-Net together, we could actually observe that output images for the Enhance-Net were all white in both experiments. The Decom-Net then gets good enhancing results for itself, but the decomposition is not actually luminance and reflectance, but something else that works anyway as image enhancing.

## Bottleneck

* encontrar pares de imagenes (por eso se usan las sinteticas)
* tamaño de imagen para conv trans
*!!!!!! do we want to explain that we had to change the image input size in order to maintain the parameters through layers?*
* Wandb en algunos casos iba muy lento
* Limitaciones de VM

## Conclusions
*!!!!!! Which model is best for us? Why?*
Si tuvieramos que escoger uno para arreglar la imagen y ya esta: todo junto con o sin ienhanced=1 (da igual porque acaba dando unos).

Si tuvieramos que escoger que arregla imagenes i mantiene descomposicion teorica: por separado y conv trans

## Execution Instructions
explicar los parametros del run.sh

## App Usage
*!!!!!! TO-DO*
python app.py in folder folder application
enter local host in explorer

*Flask*
<img src="figs/app-example.png"/>