# AFA_For_Few_shot_learning

<br>
Please cite our paper if you find the code or dataset useful for your research.

[Adversarial Feature Augmentation for Cross-domain Few-shot Classification](https://arxiv.org/abs/2208.11021)
European Conference on Computer Vision (ECCV), 2022
(```)
@article{hu2022adversarial,

  title={Adversarial Feature Augmentation for Cross-domain Few-shot Classification},
  
  author={Hu, Yanxu and Ma, Andy J},
  
  journal={arXiv preprint arXiv:2208.11021},
  
  year={2022}
  
}
(```)
<br>

Yanxu Hu, Andy Ma

ECCV 2022

## Citation
If you use this code for your research, please cite our paper

## Dependencies
* Python >= 3.5
* Pytorch >= 1.2.0 and torchvision (https://pytorch.org/)

## Datasets:
Refers to CDFSL-ATA(https://github.com/Haoqing-Wang/CDFSL-ATA)

## Train
### 1. train the baseline

```
python train.py --model ResNet10 --method GNN --n_shot 5 --name GNN_5s --train_aug
python train.py --model ResNet10 --method TPN --n_shot 5 --name TPN --train_aug
```

### 2.Train meta-learning models with feature-wise transformations.

```
python train_FT.py --model ResNet10 --method GNN --n_shot 5 --name GNN_FWT_5s --train_aug
python train_FT.py --model ResNet10 --method TPN --n_shot 5 --name TPN_FWT --train_aug
```

### 3.Explanation-guided train meta-learning models.

```
python train.py --model ResNet10 --method GNNLRP --n_shot 5 --name GNN_LRP_5s --train_aug
python train.py --model ResNet10 --method RelationNetLRP --n_shot 5 --name RelationNet_LRP --train_aug
```

### 4.Train meta-learning models with Adversarial Task Augmentation.

```
python train_ATA.py --model ResNet10 --method GNN --max_lr 80. --T_max 5 --prob 0.5 --n_shot 5 --name GNN_ATA_5s --train_aug
python train_ATA.py --model ResNet10 --method TPN --max_lr 20. --T_max 5 --prob 0.6 --n_shot 5 --name TPN_ATA --train_aug
```

### 5.train with AFA

```
python train_ND.py --model ResNet10 --method GNN --n_shot 5 --name GNN_ND --train_aug
python train_ND.py --model ResNet10 --method TPN --n_shot 5 --name TPN_ND --train_aug
```

### 6.Ablation experiments

```
python train_NND.py --model ResNet10 --method GNN --n_shot 5 --name GNN_ND --train_aug # 'worst-case feature distribution'
python train_nonlin.py --model ResNet10 --method GNN --n_shot 5 --name GNN_ND --train_aug #non-linear transformation
```

## Evaluation and Fine-tuning

### 1.Test the trained model on the unseen domains.

```
python test.py --dataset cub --n_shot 5 --model ResNet10 --method GNN --name GNN_ND
python test.py --dataset cub --n_shot 5 --model ResNet10 --method GNN --name GNN_ATA_5s
```

### 2.Fine-tuning with linear classifier.
To get the results of traditional pre-training and fine-tuning, run

```
python finetune.py --dataset cub --n_shot 5 --finetune_epoch 50 --model ResNet10 --name model_name
```

### 3.Fine-tuning the meta-learning models.

```
python finetune_ml.py --dataset cub --method GNN --n_shot 5 --finetune_epoch 50 --model ResNet10 --name model_name
```

## Note

- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot), [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot), [CDFSL-ATA](https://github.com/Haoqing-Wang/CDFSL-ATA), [cdfsl-benchmark](https://github.com/IBM/cdfsl-benchmark), [few-shot-lrp-guided](https://github.com/SunJiamei/few-shot-lrp-guided) and [TPN-pytorch](https://github.com/csyanbin/TPN-pytorch).
- The dataset, model, and code are for non-commercial research purposes only.
- You only need a GPU with 11G memory for training and fine-tuning all models.
