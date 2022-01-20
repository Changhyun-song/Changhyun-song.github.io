---
title: "Computer Visions"
permalink: /vision/
toc_sticky: true
toc_ads : true
layout: single
---
  


---

#### 🖼 **Generative Adversarial Networks : Paper Review** ([Github](https://github.com/Changhyun-song/Gan-Papers-Followup))
   

<span style='background-color: #E5EBF7;'> **GAN Basics** </span>

- `GAN`: Generative Adversarial Networks (NIPS 2014) : [arxiv](https://arxiv.org/abs/1406.2661)
- `DCGAN`: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (ICLR 2016)  : [arxiv](https://arxiv.org/abs/1511.06434)

<span style='background-color: #E5EBF7;'> **Conditional GAN** </span>

- `CGAN`: Conditional Generative Adversarial Nets (2014) : [arxiv](https://arxiv.org/abs/1411.1784)
- `ACGAN`: Conditional Image Synthesis With Auxiliary Classifier GANs (ICML 2017) : [arxiv](https://arxiv.org/abs/1610.09585)

- **Supervised Approach** 

  - `Pix2Pix`: Image-to-Image Translation with Conditional Adversarial Networks (CVPR 2017) : [arxiv](https://arxiv.org/abs/1611.07004)

  - `GAN Dissection`: Visualizing and Understanding Generative Adversarial Networks (ICLR 2019) : [arxiv](https://arxiv.org/abs/1811.10597)

  - `SPADE`: Semantic Image Synthesis with Spatially Adaptive Normalization (CVPR 2019) : [arxiv](https://arxiv.org/abs/1903.07291)

- **Unsupervised Approach** 

  - `CycleGAN`: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017) : [arxiv](https://arxiv.org/abs/1703.10593)

  - `FUNIT`: Few-Shot Unsupervised Image-to-Image Translation (ICCV 2019) : [arxiv](https://arxiv.org/abs/1905.01723)

  - `COCO-FUNIT`: Few-Shot Unsupervised Image Translation with a Content Conditioned Style Encoder (ECCV 2020) : [arxiv](https://nvlabs.github.io/COCO-FUNIT/) 
  - `HiGAN`: Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis (IJCV 2020) : [arxiv](https://arxiv.org/abs/1911.09267), [project page](https://genforce.github.io/higan/)

- **Multi Domain**
  - `BicycleGAN`: Toward Multimodal Image-to-Image Translation (NIPS 2017) : [arxiv](https://arxiv.org/abs/1711.11586) 

  - `StarGAN`: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (CVPR 2018) : [arxiv](https://arxiv.org/abs/1711.09020)

    - `StarGAN v2`: Diversity Image Synthesis for Multiple Domains (CVPR 2020) : [arxiv](https://arxiv.org/abs/1912.01865) 

  - `MUNIT` : Multi-Modal Unsupervised Image-to-Image Translation (ECCV 2018) : [arxiv](https://arxiv.org/abs/1804.04732)

  


<span style='background-color: #E5EBF7;'> **GAN Architecture** </span>

- `PGGAN`: Progressive Growing of GANs for Improved Quality, Stability, and Variation (ICLR 2018) : [arxiv](https://arxiv.org/abs/1710.10196)

- `StyleGAN`: A Style-Based Generator Architecture for Generative Adversarial Networks (CVPR 2019) : [arxiv](https://arxiv.org/abs/1812.04948)

  - `StyleGAN v2`: Analyzing and Improving the Image Quality of StyleGAN (CVPR 2020) : [arxiv](https://arxiv.org/abs/1912.04958)

  - `StyleGAN-ADA`: Training Generative Adversarial Networks with Limited Data (NeurlPS 2020) : [arxiv](https://arxiv.org/abs/2006.06676)

  - `StyleGAN v3`: Alias-Free Generative Adversarial Networks (NeurIPS 2021) : ([arxiv](https://arxiv.org/abs/2106.12423), [code](https://github.com/NVlabs/stylegan3), [project](https://nvlabs.github.io/alias-free-gan/))

- `BigGAN`: Large Scale GAN Training for High Fidelity Natural Image Synthesis (2019) : [arxiv](https://arxiv.org/abs/1809.11096), [code](https://github.com/ajbrock/BigGAN-PyTorch)


<span style='background-color: #E5EBF7;'> **Text-to-Image** </span>

- Generative Adversarial Text to Image Synthesis (ICML 2016) : [arxiv](https://arxiv.org/abs/1605.05396)


- `TediGAN`: Text-Guided Diverse Face Image Generation and Manipulation (CVPR 2021) : [arxiv](https://arxiv.org/abs/2012.03308)

- `StyleCLIP`: Text-Driven Manipulation of StyleGAN Imagery (arXiv 2021) : [arxiv](https://arxiv.org/abs/2103.17249)

- `DALLE`: Zero-Shot Text-to-Image Generation (ICML 2021) : [arxiv](https://arxiv.org/abs/2102.12092), [project page](https://openai.com/blog/dall-e/)
- Paint by Word (2021) : [arxiv](https://arxiv.org/abs/2103.10951)

<span style='background-color: #E5EBF7;'> **Improved Training Techniques** </span>

- `SS-GAN`: Self-Supervised GANs via Auxiliary Rotation Loss (CVPR 2019) : [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Self-Supervised_GANs_via_Auxiliary_Rotation_Loss_CVPR_2019_paper.pdf)

- `CR-GAN`: Consistency Regularization for Generative Adversarial Networks (ICLR 2020) : [arxiv](https://arxiv.org/abs/1910.12027)

- `ICR-GAN`: Improved Consistency Regularization for GANs (AAAI 2021) : [arxiv](https://arxiv.org/abs/2002.04724)


<span style='background-color: #E5EBF7;'> **GAN Inversion** </span>

1. **Latent Optimization**
   - `Image2stylegan`: How to embed images into the stylegan latent space? (ICCV 2019) : [arxiv](https://arxiv.org/abs/1904.03189)

   - `Image2stylegan++`: How to edit the embedded images? (CVPR 2020) : [arxiv](https://arxiv.org/abs/1911.11544)

   - `StyleFlow`: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (ACM TOG 2021) : [arxiv](https://arxiv.org/abs/2008.02401), [project page](https://rameenabdal.github.io/StyleFlow/)

   - `BDInvert`: GAN Inversion for Out-of-Range Images with Geometric Transformations (ICCV 2021) : [arxiv](https://arxiv.org/abs/2108.08998), [code](https://github.com/kkang831/BDInvert_Release)

2. **Encoder**
   - `ALAE`: Adversarial latent autoencoders (CVPR 2020) : [arxiv](https://arxiv.org/abs/2004.04467), [code](https://github.com/podgorskiy/ALAE)
   
   - `pSp`: Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation (CVPR 2021) : [arxiv](https://arxiv.org/abs/2008.00951)

3. **Hybrid approach**
   - `stylegan-encoder` : [code](https://github.com/pbaylies/stylegan-encoder)

   - `IdInvert` : In-Domain GAN Inversion for Real Image Editing (ECCV 2020) : [arxiv](https://arxiv.org/abs/2004.00049)
   
<span style='background-color: #E5EBF7;'> **Disentangled Manipulation** </span>

- `GANSpace`: Discovering Interpretable GAN Controls (NeurIPS 2020) : [arxiv](https://arxiv.org/abs/2004.02546), [code](https://github.com/harskish/ganspace)
- `GAN-Latent-Discovery`: Unsupervised Discovery of Interpretable Directions in the GAN Latent Space (2020) : [arxiv](https://arxiv.org/abs/2002.03754), [code](https://github.com/anvoynov/GANLatentDiscovery)

- `Editing in style`: Uncovering the Local Semantics of GANs (CVPR 2020) : [arxiv](https://arxiv.org/abs/2004.14367), [code](https://github.com/IVRL/GANLocalEditing)

- `HiGAN`: Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis (IJCV 2020) : [arxiv](https://arxiv.org/abs/1911.09267), [project page](https://genforce.github.io/higan/)

- `InterFaceGAN`: Interpreting the Latent Space of GANs for Semantic Face Editing (CVPR 2020) : [arxiv](https://arxiv.org/abs/1907.10786), [project page](https://genforce.github.io/interfacegan/)

- `CD3D`: Cross-Domain and Disentangled Face Manipulation with 3D Guidance (2021) : [arxiv](https://arxiv.org/abs/2104.11228), [code](https://github.com/cassiePython/cddfm3d)

- `GHFeat`: Generative Hierarchical Features from Synthesizing Images (CVPR 2021) : [arxiv](https://arxiv.org/abs/2007.10379), [project page](https://genforce.github.io/ghfeat/)

- `StyleSpace`: StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation (2021) : [arxiv](https://arxiv.org/abs/2011.12799), [code](https://github.com/xrenaa/StyleSpace-pytorch)

- `StyleFlow`: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (ACM TOG 2021) : [arxiv](https://arxiv.org/abs/2008.02401), [project page](https://rameenabdal.github.io/StyleFlow/)

- `Hessian Penalty`: A weak prior for unsupervised disentanglement (ECCV 2020) : [arxiv](https://arxiv.org/abs/2008.10599), [project page](https://www.wpeebles.com/hessian-penalty)

- `StyleGAN of All Trades`: Image Manipulation with Only Pretrained StyleGAN (arxiv 2021) : [arxiv](https://arxiv.org/abs/2111.01619), [code](https://github.com/mchong6/SOAT)


<span style='background-color: #E5EBF7;'> **Image Editing** </span>

- `StyleCLIP`: Text-Driven Manipulation of StyleGAN Imagery (arXiv 2021) : [arxiv](https://arxiv.org/abs/2103.17249), [code](https://github.com/orpatashnik/StyleCLIP)

- `sefa`: Closed-Form Factorization of Latent Semantics in GANs (CVPR 2021) : [arxiv](https://arxiv.org/abs/2007.06600)

- `EigenGAN`: Layer-Wise Eigen-Learning for GANs : [arxiv](https://arxiv.org/abs/2104.12476), [code](https://github.com/bryandlee/eigengan-pytorch)

- `StyleMapGAN`: Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing (CVPR 2021) : [arxiv](https://arxiv.org/abs/2104.14754), [code](https://github.com/naver-ai/StyleMapGAN)

- `SEAN`: Image Synthesis with Semantic Region-Adaptive Normalization (CVPR 2020) : [arxiv](https://arxiv.org/abs/1911.12861), [code](https://github.com/ZPdesu/SEAN)
- `CDDFM3D`: Cross-Domain and Disentangled Face Manipulation with 3D Guidance (2021) : [arxiv](https://arxiv.org/abs/2104.11228), [code](https://github.com/cassiePython/cddfm3d), [project](https://cassiepython.github.io/sigasia/cddfm3d.html)

- `MocoGAN-HD`: A Good Image Generator Is What You Need for High-Resolution Video Synthesis (ICLR 2021) : [arxiv](https://arxiv.org/abs/2104.15069), [code](https://github.com/snap-research/MoCoGAN-HD), [project](https://bluer555.github.io/MoCoGAN-HD/)

- `StyleGAN-NADA`: CLIP-Guided Domain Adaptation of Image Generators (arxiv 2021) : [arxiv](https://arxiv.org/abs/2108.00946), [project](https://stylegan-nada.github.io/), [code](https://arxiv.org/abs/2108.00946)

- `HyperStyle`: StyleGAN Inversion with HyperNetworks for Real Image Editing (arxiv 2021) : [arxiv](https://arxiv.org/abs/2111.15666), [code](https://github.com/yuval-alaluf/hyperstyle)


<span style='background-color: #E5EBF7;'> **Webtoon/Anime GAN & Image Blending** </span>

- `Cartoon-StylaGAN`: Fine-tuning StyleGAN2 for Cartoon Face Generation (arxiv 2021) : [arxiv](https://arxiv.org/abs/2106.12445), [review](https://happy-jihye.github.io/gan/gan-21/), [code](https://github.com/happy-jihye/Cartoon-StyleGAN)
- `BlendGAN`: Implicitly GAN Blending for Arbitrary Stylized Face Generation (NeurIPS 2021) : [arxiv](https://arxiv.org/abs/2110.11728), [project](https://onion-liu.github.io/BlendGAN/), [code](https://github.com/onion-liu/BlendGAN)
- `HifiFace`: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping (IJCAI 2021) : [arxiv](https://arxiv.org/abs/2106.09965)
- `AnimeGAN`: A Novel Lightweight GAN for Photo Animation (ISICA 2019)
  - `AnimeGANv2` (2021) : [project](https://tachibanayoshino.github.io/AnimeGANv2/), [code](https://github.com/bryandlee/animegan2-pytorch)
- `StyleGAN of All Trades`: Image Manipulation with Only Pretrained StyleGAN (arxiv 2021) : [arxiv](https://arxiv.org/abs/2111.01619), [code](https://github.com/mchong6/SOAT)


<span style='background-color: #E5EBF7;'> **Super Resolution** </span>

- `BSRGAN`: Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (ICCV 2021) : [arxiv](https://arxiv.org/abs/2103.14006), [code](https://github.com/cszn/BSRGAN)

- `Real-ESRGAN`: Training Real-World Blind Super-Resolution with Pure Synthetic Data (ICCVW 2021): [arxiv](https://arxiv.org/abs/2107.10833), [code](https://github.com/xinntao/Real-ESRGAN)


<span style='background-color: #E5EBF7;'> **3D GAN & Rendering** </span>

- `HoloGAN`: Unsupervised learning of 3D representations from natural images (ICCV 2019): [paper](https://arxiv.org/abs/1904.01326), [code](https://github.com/thunguyenphuoc/HoloGAN)
- `CDDFM3D`: Cross-Domain and Disentangled Face Manipulation with 3D Guidance (2021) : [arxiv](https://arxiv.org/abs/2104.11228), [code](https://github.com/cassiePython/cddfm3d), [project](https://cassiepython.github.io/sigasia/cddfm3d.html)

- `SofGAN`: A Portrait Image Generator with Dynamic Styling (arxiv 2021): [arxiv](https://arxiv.org/abs/2007.03780), [project](https://apchenstu.github.io/sofgan/), [code](https://github.com/apchenstu/sofgan)

- `pi-GAN`: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis (CVPR 2021): [paper](https://arxiv.org/pdf/2012.00926.pdf), [project](https://marcoamonteiro.github.io/pi-GAN-website/), [code](https://marcoamonteiro.github.io/pi-GAN-website/)
- `StyleGANRender`: Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering (ICLR 2021) : [arxiv](https://arxiv.org/abs/2010.09125), [project page](https://nv-tlabs.github.io/GANverse3D/)

- `StyleNeRF`: A Style-based 3D Aware Generator for High-resolution Image Synthesis (ICLR 2022): [paper](https://openreview.net/forum?id=iUuzzTMUw9K)

  - `NeRF`: Representing Scenes as Neural Radiance Fields for View Synthesis (ECCV 2020) : [arxiv](https://arxiv.org/abs/2003.08934), [project](https://www.matthewtancik.com/nerf)

- `CIPS-3D`: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis (arxiv 2021): [arxiv](https://arxiv.org/abs/2110.09788), [code](https://github.com/PeterouZh/CIPS-3D)

- `PIRenderer`: Controllable Portrait Image Generation via Semantic Neural Rendering (ICCV 2021) : [arxiv](https://arxiv.org/abs/2109.08379), [project](https://renyurui.github.io/PIRender_web/), [code](https://github.com/RenYurui/PIRender)

---

## Reference 
- https://github.com/eriklindernoren/PyTorch-GAN