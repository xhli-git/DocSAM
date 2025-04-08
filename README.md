# DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning (CVPR 2025)
## Features
- A uified model for various document segmentation tasks and datasets.
- A pretrained model that can be seemlessly migrate to new segmentation tasks.


## Introduction

DocSAM makes two significant improvements compared to existing methods. First, it transforms various document image segmentation tasks into a unified paradigm of mask-based instance segmentation and semantic segmentation. Second, it embeds class names into semantic queries, which not only serve as prompts to guide the model in identifying specific types of regions to segment but also function as class prototypes that instance queries depend on for classification.

After mixed pretraining on heterogeneous tasks and datasets, the DocSAM model can be directly used as a versatile document segmenter or as a pre-trained model that can be seamlessly fine-tuned using task-specific datasets without any specialized modifications, such as adding or replacing a linear classification layer. This merit of DocSAM endows it with the potential for continual and incremental learning. The framework of DocSAM is shown in the figure below. 


![DocSAM](figures/DocSAM.png)


## Installation
```
conda create --name DocSAM python=3.11 -y
conda activate DocSAM

git clone git@github.com:xhli-git/DocSAM.git
cd DocSAM
pip install -r requirments.txt
```

## Getting Started
### Preparing Datasets
All dataset annotations must be converted to the unified COCO format. We provide a [demo dataset](https://drive.google.com/file/d/1gvfco5zyRDASGO2BOYCjZuRxbN7MHhsT/view?usp=drive_link) to illustrate the structure and format of the datasets used during training and testing. Please download it and place it in the designated 'data' folder. You can prepare your own data according to the structure and guidelines outlined in this demo dataset.

### Preparing Weights
DocSAM is build upon Mask2Former and Sentence-BERT, please download their pretrained weights from huggingface and place them in the designated 'pretrained_model' folder.
- Download [mask2former-swin-base-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic) and place it in the designated './pretrained_model/mask2former' folder.
- Download [mask2former-swin-large-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic) and place it in the designated './pretrained_model/mask2former' folder.
- Download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and place it in the designated './pretrained_model/sentence' folder.
- Download pretrained DocSAM weights from the Model Zoo below and place them in the designated './pretrained_model' folder.

### Run Inference Demo
Make sure the configurations in ```run_test_demo.sh``` meet your requirements and run: 
```
sh run_test_demo.sh
```
The predict outputs can be found in the 'outputs' folder.

### Run Training Demo
Make sure the configurations in ```run_train_demo.sh``` meet your requirements and run: 
```
sh run_train_demo.sh
```
The trained model weights can be found in the "snapshots" folder.

### More Scripts
In the 'scripts' folder, we provide some scripts helpful for training, testing and inference. You can select or create your own script based on them. We also provide training scripts for curriculum learning, which progressively incorporates new datasets into the training process to accelerate model convergence. 

### Full Size Training
Make sure you have prepared your datasets according to the [demo dataset](https://drive.google.com/file/d/1gvfco5zyRDASGO2BOYCjZuRxbN7MHhsT/view?usp=drive_link), and the configurations in ```run_train_curriculum_large_all_dataset.sh``` meet your requirements, then run:
```
sh run_train_curriculum_large_all_dataset.sh
```

### Pycocotools Modifications
By default, ```pycocotools``` only displays the mean Average Precision (mAP) averaged across all classes and limits the maximum number of objects per image to 100. If you need more detailed information, such as the AP for each individual class, and your images contain more than 100 objects, modifications to the library's source code are necessary. Below, we highlight the differences between the modified code and the original code.

For coco.py:
```python
84a85
>             print("category names: {}".format([e["name"] for e in sorted(dataset["categories"], key=lambda x: x["id"])]))
426a428,429
>             # print("###############################################")
>             # print(segm)

```

For cocoeval.py:
```python
456c456,469
<             print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
---
>             #print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
>            
>             category_dimension = 1 + int(ap)
>             if s.shape[category_dimension] > 1:
>                 iStr += ", per category = {}"
>                 mean_axis = (0,)
>                 if ap == 1:
>                     mean_axis = (0, 1)
>                 per_category_mean_s = np.mean(s, axis=mean_axis).flatten()
>                 with np.printoptions(precision=3, suppress=True, sign=" ", floatmode="fixed"):
>                     print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s, per_category_mean_s))
>             else:
>                 print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s, ""))
> 
460c473,474
<             stats[0] = _summarize(1)
---
>             ### stats[0] = _summarize(1)
>             stats[0] = _summarize(1, maxDets=self.params.maxDets[2])

```


## Model Zoo

Please see [Model Zoo](MODELZOO.md).


## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of DocSAM is licensed under a [MIT License](LICENSE).

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).


## Citation
If you find DocSAM helpful for your research or wish to refer to the baseline results published in our paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{li2025docsam,
  title={DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning},
  author={Xiao-Hui Li and Fei Yin and Cheng-Lin Liu},
  journal={CVPR},
  year={2025}
}
```

## Acknowledgement

Code is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former).

Pretrained weights of [Mask2Former](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) are downloaded from huggingface.
