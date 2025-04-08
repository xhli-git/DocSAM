import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.set_printoptions(precision=4, threshold=np.inf, edgeitems=None, linewidth=1000, profile=None, sci_mode=False)

from transformers import logging
from models.mask2former import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
logging.set_verbosity_error()
 


__all__ = [
    "DocSAM",
]


# DocSAM
class DocSAM(nn.Module):
    """
    Document Segment Anything Model.
    """
    
    def __init__(self, model_size="base"):
        super(DocSAM, self).__init__()

        # Set your pretrained model path here.
        if model_size=="base":
            mask2former_path = "./pretrained_model/mask2former/facebook-mask2former-swin-base-coco-panoptic/"
        elif model_size=="large":
            mask2former_path = "./pretrained_model/mask2former/facebook-mask2former-swin-large-coco-panoptic/"
        else:
            mask2former_path = "./pretrained_model/mask2former/facebook-mask2former-swin-base-coco-panoptic/"
        print("mask2former_path:", mask2former_path)
        
        sentence_path = "./pretrained_model/sentence/all-MiniLM-L6-v2"
        print("sentence_path:", sentence_path)
        
        # Set your configurations here.
        self.config = Mask2FormerConfig.from_pretrained(mask2former_path)
        self.config.sentence_path = sentence_path

        self.config.num_queries = 900
        self.config.query_selection = True
        self.config.min_num_queries = 100
        self.config.max_num_queries = 900
        
        self.config.class_weight = 1.0
        self.config.sml1_weight  = 1.0
        self.config.diou_weight  = 1.0
        self.config.dice_weight  = 5.0
        self.config.focal_weight = 5.0
        
        self.config.init_std = 0.02 # default: 0.02
        
        self.config.encoder_layers = 6
        self.config.decoder_layers = 4
        self.config.num_levels = 4
        self.config.is_multi_scale = True

        self.config.hidden_dim = 256
        self.config.feature_size = 256
        self.config.mask_feature_size = 256
        self.config.feature_strides = [4, 8, 16, 32]
        
        # Load weights from pretrained mask2former model.
        # self.mask2former = Mask2FormerForUniversalSegmentation(self.config)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_path, config=self.config, ignore_mismatched_sizes=True)
        # self.mask2former = self._load_pretrained_visual_backbone_paras(self.mask2former, "./snapshots_mim/last_model.pth")


    def _load_pretrained_visual_backbone_paras(self, model, model_path):
        """
        Loads pretrained parameters for the visual backbone of the model.

        Args:
            model: The current model instance.
            model_path (str): Path to the pretrained model weights.

        Returns:
            model: Updated model with loaded parameters.
        """
        
        pre_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
        pre_dict = {"model." + k: v for k, v in pre_dict.items()}
        cur_dict = {k: v for k, v in model.named_parameters() if "pixel_level_module" in k}
        #print("cur_dict:", cur_dict.keys())

        matched_dict = {}
        unmatched_keys = []
        for k in cur_dict.keys():
            if k in pre_dict and cur_dict[k].size() == pre_dict[k].size():
                matched_dict[k] = pre_dict[k]
            else:
                unmatched_keys.append(k)
                
        if unmatched_keys:
            print("Unmatched keys in current model:", unmatched_keys)

        model.load_state_dict(matched_dict, strict=False)
        print("Pretrained model loaded!!!", model_path)

        return model
    

    # random image masking
    def _image_mask(self, x, patch_size=16, mask_ratio=0.5, mask_method=0):
        """
        Applies random masking on the input images.

        Args:
            x (Tensor): Input image tensor.
            patch_size (int): Size of each patch. Default is 16.
            mask_ratio (float): Ratio of patches to be masked. Default is 0.5.
            mask_method (int): Method of masking; can be 0 (patchwise), 1 (channelwise), or 2 (mix). Default is 0.

        Returns:
            Tuple[Tensor, Tensor]: The masked image and the corresponding mask tensor.
        """
        
        N,C,H,W = x.size()

        patch_num = (math.ceil(H/patch_size), math.ceil(W/patch_size))
        if mask_method == 0:    # patchwise
            mask = torch.rand((N, 1, patch_num[0], patch_num[1]), device=x.device).tile((1,C,1,1))
            mask = mask.view(N,C,-1).argsort(dim=2).view(N,C,patch_num[0],patch_num[1])
            mask = mask <= mask_ratio * (patch_num[0] * patch_num[1] - 1)
        elif mask_method == 1:  # channelwise
            mask = torch.rand((N, C, 1, 1), device=x.device).tile((1,1,patch_num[0],patch_num[1]))
            mask = mask.argsort(dim=1)
            #mask = mask <= mask_ratio * (C - 1)
            mask = mask <= random.randint(0, C - 2)
        elif mask_method == 2:  # mix
            mask = torch.rand((N, C, patch_num[0], patch_num[1]), device=x.device)
            mask = mask.view(N,C,-1).argsort(dim=2).view(N,C,patch_num[0],patch_num[1])
            mask = mask <= mask_ratio * (patch_num[0] * patch_num[1] - 1)
        else:                   # default: patchwise
            mask = torch.rand((N, 1, patch_num[0], patch_num[1]), device=x.device).tile((1,C,1,1))
            mask = mask.view(N,C,-1).argsort(dim=2).view(N,C,patch_num[0],patch_num[1])
            mask = mask <= mask_ratio * (patch_num[0] * patch_num[1] - 1)

        mask = F.interpolate(mask.float(), scale_factor=(patch_size, patch_size), mode='nearest')
        mask = mask[:,:,0:H,0:W].int()
        x[mask==1] = 0

        return x, mask


    def forward(self, batch):
        """
        Defines the computation performed at every call. Receives a batch of inputs and passes them through the model.

        Args:
            batch (dict): Dictionary containing the following keys:
                - pixel_values (Tensor): Image pixel values.
                - pixel_mask (Tensor): Pixel mask.
                - instance_masks (List[Tensor]): Instance masks.
                - instance_bboxes (List[Tensor]): Bounding boxes for instances.
                - instance_labels (List[Tensor]): Labels for instances.
                - semantic_masks (List[Tensor]): Semantic masks.
                - class_names (List[List[str]]): Names of classes.
                - coco_datas (Dict): Additional data related to COCO format.
                - img_bboxes (Tensor): Image bounding boxes.
                - datasets (List[str]): Dataset names of images.
                - names (List[str]): Image names.

        Returns:
            outputs (Mask2FormerForUniversalSegmentationOutput): Model output containing predictions and losses.
        """
        
        pixel_values, pixel_mask, instance_masks, instance_bboxes, instance_labels, semantic_masks, class_names, coco_datas, image_bboxes, dataset_names, image_names = \
            batch["pixel_values"], batch["pixel_mask"], batch["instance_masks"], batch["instance_bboxes"], batch["instance_labels"], \
            batch["semantic_masks"], batch["class_names"], batch["coco_datas"], batch["image_bboxes"], batch["dataset_names"], batch["image_names"]

        outputs = self.mask2former(pixel_values=pixel_values,
                                   pixel_mask=pixel_mask,
                                   image_bboxes=image_bboxes,
                                   instance_masks=instance_masks, 
                                   instance_bboxes=instance_bboxes, 
                                   instance_labels=instance_labels, 
                                   semantic_masks=semantic_masks, 
                                   class_names=class_names,)
        
        return outputs