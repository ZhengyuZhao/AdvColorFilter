# AdvCF
PyTorch code for our arXiv article:

[**A Differentiable Color Filter for Generating Unrestricted Adversarial Images**](https://arxiv.org/abs/1911.02466)

We propose Adversarial Color Filtering (AdvCF), an approach that uses a differentiable color filter to create adversarial images.
This paper validate two properties of the resulting adversarial images: 1) the photographic quality and appeal of them are maintained or enhanced, even with large perturbations introduced. 2) Higher transferabibility than conventional L<sub>p</sub> approaches. We also points out that AdvCF could be further improved if image semantics are taken into account.

## Implementation
 
### Requirements
torch>=1.1.0; torchvision>=0.3.0; tqdm>=4.31.1; pillow>=5.4.1; matplotlib>=3.0.3;  numpy>=1.16.4; 

### Download data

Run [this official script](https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/dataset/download_images.py) to download the dataset.

### Generating adversarial examples with AdvCF


### Examples

<p align="center">
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/examples_9.PNG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/examples_7.PNG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/examples_8.PNG" width='1000'alt>
  <em>Fig. 1: Adversarial images generated by CW (3x100, &kappa;=20), BIM (&epsilon;=8), and our AdvCF (s=1/64, &lambda;=5).</em>
</p>

<br /><br />

<p align="center">
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_2.JPG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_3.JPG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_4.JPG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_6.JPG" width='1000'>
  <img src="https://github.com/ZhengyuZhao/AdvCF/blob/master/figure/ex_7.JPG" width='1000'alt>
  <em>Fig. 2: More examples generated by our AdvCF (s=1/64, &lambda;=5).</em>
</p>
