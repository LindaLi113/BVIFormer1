# BVIFormer: A Binocular-Vision-Inspired Transformer with Binocular Competitive Fusion for Single-Image Restoration

## Welcome to our workplace~💕👍

Human vision relies on the interplay between foveal processing and binocular integration to robustly perceive structures in degraded scenes. Motivated by this observation, we revisit the image restoration problem from a binocular-vision perspective. We model binocular integration as a competitive attention mechanism and instantiate this concept in BVIFormer, a binocular-vision-inspired restoration network with a multi-scale architecture centered on a Binocular Competitive Fusion (BCF) module. BCF explicitly models competition between the two views to sharpen informative structures and suppress degradations. At each stage, a dual-branch block couples an image embedding module, Transformer blocks, and the BCF module to mimic the binocular perception and integration pathways. We apply this framework to single-image dehazing and deraining tasks. On the Dense-Haze and SOTS-Outdoor benchmarks, BVIFormer achieves 17.30 dB and 37.56 dB PSNR, respectively, surpassing the previous best results by 0.86 dB and 0.14 dB. On Rain1400 and Test2800, it attains 33.82 dB and 34.20 dB PSNR, respectively. Extensive experiments on synthetic and real-world benchmarks show consistent gains under heavy haze and severe rain streaks. Code and datasets are available at \nolinkurl{https://github.com/LindaLi113/BVIFormer}.

 We have submitted our paper to Scientific Report. Coming up soon......

<img width="1044" height="430" alt="figure 1" src="https://github.com/user-attachments/assets/eb027c74-84e1-414c-b62c-2f31a95d3425" />
 
<img width="6532" height="2836" alt="figure2" src="https://github.com/user-attachments/assets/f50b304f-6d8b-4934-b032-3edd6289e67b" />

<img width="6066" height="3315" alt="figure 3" src="https://github.com/user-attachments/assets/c28d673d-d7e1-4c16-95f6-72525f12757c" />

<img width="3175" height="1789" alt="figure 4" src="https://github.com/user-attachments/assets/0e28b1a3-de00-45fb-a82d-d3faf8ae70ff" />

<img width="1978" height="679" alt="figure 5" src="https://github.com/user-attachments/assets/e373508b-2f72-4747-b84f-0498f02721b5" />

<img width="1183" height="547" alt="figure 6" src="https://github.com/user-attachments/assets/234d388d-299d-4e5c-bb9c-313aded1bad6" />

<img width="2152" height="1439" alt="figure 7" src="https://github.com/user-attachments/assets/1b86edb0-106f-465b-9774-977303afeb48" />



