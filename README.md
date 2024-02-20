## Official Pytorch implementation of "Patch-wise Graph Contrastive Learning for Image Translation" (AAAI 2024)
[Chanyong Jung](https://sites.google.com/view/jcy132), [Gihyun Kwon](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html) 

Link: https://arxiv.org/abs/2312.08223

![gcl](https://github.com/jcy132/PatchGCL/assets/52989204/af5ed888-cc62-4657-b348-bda295398f99)
* We provide the pretrained model: [horse-to-zebra](https://drive.google.com/file/d/1T-gaGGrg7mVytUmlCeZ-ExbrIa9D8iCF/view?usp=sharing)
* For the training from scratch (ex. horse2zebra dataset):
```
python train.py --name [folder-name] --dataroot [path-to-data] \
--lambda_GNN 0.1 --num_hop 2 --gnn_idt --nonzero_th 0.1 \
--pooling_num 1 --pooling_ratio '1,0.5' --down_scale 4 \
--gpu_ids 0
```
* For evaluation:
```
python test.py --dataroot [path-to-dataset] --name [experiment-name] \
--CUT_mode CUT --phase test --epoch [epoch-for-test] --num_test [test-size]

python -m pytorch_fid [path-to-output] [path-to-input]
```


### Cite
```
@article{jung2023patch,
  title={Patch-wise Graph Contrastive Learning for Image Translation},
  author={Jung, Chanyong and Kwon, Gihyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2312.08223},
  year={2023}
}
```

### Acknowledgement
Our source code is based on the official implementation of [HnegSRC](https://github.com/jcy132/Hneg_SRC) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation). 
