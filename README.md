# [Generative Dataset Distillation Based on Diffusion Model](https://arxiv.org/abs/2408.08610)


## [ECCV 2024 The First Dataset Distillation Challenge](https://www.dd-challenge.com/) Track 2 Third Place


### Environment

- Python >=3.9
- Pytorch >= 1.12.1
- Torchvision >= 0.13.1
- Diffusers == 0.29.2


### Generate Images

- For CIFAR100, run:
``````python
python submit_cifar100.py
``````

- For TinyImagenet, run:
``````python
python submit_tinyimagenet.py
``````

### Note

- IPC needs to be set as a multiple of 5.


### Evaluation

- Follow the [official repository](https://github.com/DataDistillation/ECCV2024-Dataset-Distillation-Challenge)


## Citation
If you find this paper useful for your research, please use the following BibTeX entry.
```
@inproceedings{su2024diffusion,
  title={Generative Dataset Distillation Based on Diffusion Model},
  author={Su, Duo and Hou, Junjie and Li, Guang and Togo, Ren and Song, Rui and Ogawa, Takahiro and Haseyama, Miki},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV), Workshop},
  year={2024}
}
```
