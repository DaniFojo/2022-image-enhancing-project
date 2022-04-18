# LOW-LIGHT IMAGE ENHANCEMENT

In this project we implement RetinexNet with PyTorch based on this paper:

[Deep Retinex Decomposition for Low-Light Enhancement](https://paperswithcode.com/paper/deep-retinex-decomposition-for-low-light)

We know that insufficient lighting can significantly degrade the visibility of images and that there are many reconstruction/enhancing techniques in order to get better quality.

<figure>
<img src="figs/low-normal-light-images.png"/>
<figcaption align = "left"> 
Several Low-Light / Normal images  
</figcaption>
</figure>

We chose Retinex model as an effective tool for low-light image enhancement. One of the main reasons was that it assumes that observed images can be decomposed into their reflectance and illumination, a theory that was postulated by Edwin Herbert Land which denies trichromatic Newton color theory and states that color is a brain active composition done by comparison.

<figure>
<img src="figs/color_theory.png"/>
<figcaption align = "left"> 
A = B (same color)
</figcaption>
</figure>

[¿Qué es el color? La teoría Retinex de Land](http://opticaluzycolor.blogspot.com/2011/03/que-es-el-color-la-teoria-retinex-de.html?m=1)

## Datasets

We used both datasets already used in the original paper:

Dataset 1: [LOL (LOw-Light dataset)](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)  
500 real images (485 to train and 15 to test), 400x600px.

Dataset 2: [Synthetic pairs](https://drive.google.com/file/d/1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F/view)  
1250 images (1000 to train and 250 to test), different sizes.

And we have created our own set of 10 pair of images with normal and low light to test the final model, these are some examples:

<img src="figs/readme_example.png" width="200px"/>

## Net structure
In the figure below we can find the proposed framework for Retinex-Net. The enhancement process is divided into three steps: decomposition, adjustment and reconstruction. In the decomposition step, a subnetwork Decom-Net decomposes the input image into reflectance and illumination. In the following adjustment step, an encoder-decoder based Enhance-Net brightens up the illumination. Multi-scale concatenation is introduced to adjust the illumination from multi-scale perspectives. Noise on the reflectance is also removed at this step. Finally, we reconstruct the adjusted illumination and reflectance to get the enhanced result.

<img src="figs/retinexnet.png" width="600px"/>

## Model training

We have trained the net in mainly 4 different ways attending the following aspects:

* Decom-Net and Enhance-Net separately
    * With convolutional + resize layers in Enhance-Net
    * With convolutional transposed layers in Enhance-Net
* Decom-Net and Enhance-Net together
    * Training both nets
    * Ignoring Enhance-Net

In all four experiments we have trained 200 epochs and started with learning rate at 0.0001 for both nets.

<span style="color:red">
!!TO-DO!!:  

Explain motivation of 4 experiments (ienhanced = 1 because we saw white images in results, transposed because it is a better solution rather than resizing using interpolation, etc )
Explain why we didn't use denoising (original function was impossible to use and open-cv was high cost/effective)

<b>Logger</b>  
Wandb

<b>Scheduler</b>  
Step LR vs ReduceOnPlateau

<b>Platform</b>  
Google Cloud

<b>Configuration</b>  
In project folder:  
python 3.8 -m venv  
source venv/bin/activate  
pip install -r requirements.txt
</span>.


## Results
Is it possible links to 4 wandb workspaces? Either way, some screenshots here.

Which model is best for us? Why?

## App Usage
Flask