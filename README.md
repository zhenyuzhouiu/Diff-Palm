<div align="center">

# Official implementation of "Diff-Palm: Realistic Palmprint Generation with Polynomial Creases and Intra-Class Variation Controllable Diffusion Models" [CVPR2025]

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="http://arxiv.org/abs/2503.18312" target='_blank'><img src="https://img.shields.io/badge/arXiv-2503.18312-b31b1b.svg"></a>
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Ukuer/Diff-Palm">
    

</div>

This repository is the official PyTorch implementation of the CVPR paper "Diff-Palm: Realistic Palmprint Generation with Polynomial Creases and Intra-Class Variation Controllable Diffusion Models".

## Abstract
Palmprint recognition is significantly limited by the lack of large-scale publicly available datasets. 
Previous methods have adopted B\'ezier curves to simulate the palm creases, which then serve as input for conditional GANs to generate realistic palmprints.
However, without employing real data fine-tuning, the performance of the recognition model trained on these synthetic datasets would drastically decline, indicating a large gap between generated and real palmprints.
This is primarily due to the utilization of an inaccurate palm crease representation and challenges in balancing intra-class variation with identity consistency.
To address this, we introduce a polynomial-based palm crease representation that provides a new palm crease generation mechanism more closely aligned with the real distribution. 
We also propose the palm creases conditioned diffusion model with a novel intra-class variation control method.
By applying our proposed $K$-step noise-sharing sampling, we are able to synthesize palmprint datasets with large intra-class variation and high identity consistency.
Experimental results show that, for the first time, recognition models trained solely on our synthetic datasets, without any fine-tuning, outperform those trained on real datasets.
Furthermore, our approach achieves superior recognition performance as the number of generated identities increases.


## Getting Started

### Clone this repository
Clone our repo to your local machine using the following command:
```bash
git clone https://github.com/Ukuer/Diff-Palm.git
cd Diff-Palm
```

### Prerequisites
- The dependent packages are listed in `requirements.txt` file.
- Note that this diffusion model is based on [guided-diffusion](https://github.com/openai/guided-diffusion).
If you have any environment problems, you may find solutions in [guided-diffusion](https://github.com/openai/guided-diffusion) and its parent repo [improved-diffusion](https://github.com/openai/improved-diffusion).


## Training
The palm creases conditioned diffusion model are placed in `DiffModels`. To train this model, please refer to the following steps:
- prepare the palmprint ROI images.
- extract the palm crease images using [PCEM](https://github.com/Ukuer/PCE-Palm/blob/main/PCEM_numpy.py).
- place paired palmprint and palm crease directory in `palm` and `label` sub-directory respectively, as follows :
  ```
  DATASETS/
  ├── palm/
  │   ├── 1.png
  │   ├── 2.png
  │   └── ...
  └── label/
      ├── 1.png
      ├── 2.png
      └── ...
  ```
- modify the `run.sh` to meet your requirements.
- `bash run.sh` to start training. You need to kill the process manually to finish the training.
- ---
- Additionally, we have found that training with plain palmprint images would easily result in synthesizd images with color shift. A small discussion about this problem is in [issues](https://github.com/openai/guided-diffusion/issues/81). To avoid this, we opt to apply `scale.py` to scale each palmprint images.

## Inference
The infernece is two-stage. **First synthsize polynomial creases, then generate palmprints with these creases.**

### Polynomial Creases
This code is in `PolyCreases`. 
- run `syn_polypalm_mp.py` to synthesize polynomial crease images. 
- Note that each synthesized image is regarded as an identity.  

### Diffusion Models
This code is in `DiffModels`.
- download pretrained weights in [Google drive](https://drive.google.com/file/d/1vQya0fgrSh-PkFsDBi0OM89_rfSiO3u6/view?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1p6B2TmYfQCdUQdcZRztLVw?pwd=q835).
- place this pretrained file in `./checkpoint/diffusion-netpalm-scale-128`
- modify the `sample.sh` to meet your requirements.
- run `bash sample.sh` to synthsize realistic palmprint datasets.

## Datasets
TODO

## Evaluation 
TODO

## Acknowledgement

Our implementation is based on the following works: [PCE-Palm](https://github.com/Ukuer/PCE-Palm), [guided-diffusion](https://github.com/openai/guided-diffusion), [ElasticFace](https://github.com/fdbtrs/ElasticFace).


## Citation

```
@misc{jin2025diffpalm,
      title={Diff-Palm: Realistic Palmprint Generation with Polynomial Creases and Intra-Class Variation Controllable Diffusion Models}, 
      author={Jianlong Jin and Chenglong Zhao and Ruixin Zhang and Sheng Shang and Jianqing Xu and Jingyun Zhang and ShaoMing Wang and Yang Zhao and Shouhong Ding and Wei Jia and Yunsheng Wu},
      year={2025},
      eprint={2503.18312},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18312}, 
}
```
