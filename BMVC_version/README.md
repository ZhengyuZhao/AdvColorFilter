# ACE
PyTorch code for our paper:

Zhengyu Zhao, Zhuoran Liu, Martha Larson, [**"Adversarial Color Enhancement: Generating Unrestricted Adversarial Images by Optimizing a Color Filter"**](https://arxiv.org/abs/2002.01008), BMVC 2020.
<p align="center">
  <img src="https://github.com/ZhengyuZhao/ACE/blob/master/BMVC_version/Figures/figure1.PNG" width='600'>
</p>
Deep computer vision models are susceptible to adversarial perturbations.
We propose Adversarial Color Enhancement (ACE), a novel approach to generating non-suspicious adversarial images by optimizing color transformation within a parametric filter space. Because filters modify similar colors in similar ways, ACE inherently obviates the need for additional constraints. We investigate a general formulation of ACE and also a variant targeting a particular color style, i.e., that steers an image towards a specific, popular image color filter. We carry out a robustness analysis from both the attack and defense perspectives by adjusting the bounds of the color filter parameters. From the attack perspective, we provide extensive experiments on the vulnerability of image classifiers, but also explore the vulnerability of segmentation and aesthetics quality assessment algorithms. From the defense perspective, more experiments provide insight into the stability of ACE against input transformation-based defenses and show the potential of adversarial training for improving model robustness against ACE.

## Implementation
 
### Requirements
torch>=1.3.1; torchvision>=0.4.2; scipy=1.3.x (**Important!** The 1.4.x releases of scipy have resulted in greater than 100x slow-down of "stats.truncnorm.rvs" relative to the 1.3.x versions when loading Inception-V3 model in PyTorch.)
### Download data

Run [this official script](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py) to download the dataset.

### Generating adversarial examples with ACE

```
python main.py -batch_size 25 -gpu 1 -max_iterations 500 -learning_rate 0.01 -pieces 64 -search_steps 1 -initial_lambda 5
```

### Examples for two tasks (ImageNet classification and Places scene recognition)

<p align="center">
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/BMVC_version/Figures/add_1.PNG" width='800'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/BMVC_version/Figures/add_2.PNG" width='800'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/BMVC_version/Figures/add_3.PNG" width='800'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/BMVC_version/Figures/add_4.PNG" width='800'alt>
<!--   <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_7.JPG" width='1000'alt> -->

