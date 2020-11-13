## About
PyTorch code for our paper:

Zhengyu Zhao, Zhuoran Liu, Martha Larson, [**"Adversarial Robustness Against Image Color Transformation within Parametric Filter Space
"**](), under review.
<p align="center">
  <img src="https://github.com/ZhengyuZhao/ACE/blob/master/Journal_version/Figures/illustration.PNG" width='800'>
</p>

We propose Adversarial Color Enhancement (ACE), a novel approach to generating non-suspicious adversarial images by optimizing color transformation within a parametric filter space. Because filters modify similar colors in similar ways, ACE inherently obviates the need for additional constraints. We investigate a general formulation of ACE and also a variant targeting a particular color style, i.e., that steers an image towards a specific, popular image color filter. We carry out a robustness analysis from both the attack and defense perspectives by adjusting the bounds of the color filter parameters. From the attack perspective, we provide extensive experiments on the vulnerability of image classifiers, but also explore the vulnerability of segmentation and aesthetics quality assessment algorithms. From the defense perspective, more experiments provide insight into the stability of ACE against input transformation-based defenses and show the potential of adversarial training for improving model robustness against ACE.

## Implementation

### Overview

This code contains the implementations of:
 1. The proposed ACE on attacking ImageNet classifiers in ```ACE.ipynb```,
 2. The adversarial training against ACE on CIFAR-10 in the subfolder ```AdvTrain_ACE```.
 
### Requirements
torch>=1.1.0; torchvision>=0.3.0; tqdm>=4.31.1; pillow>=5.4.1; matplotlib>=3.0.3;  numpy>=1.16.4; 

