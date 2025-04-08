
import os
import timeit
import random
from tqdm import tqdm

import argparse
import math
import json
import copy
import numpy as np
np.set_printoptions(linewidth=400)
np.set_printoptions(precision=4)

import cv2
from PIL import Image as PILImage
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment

import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.ops import box_iou
from torch.nn.utils.rnn import pad_sequence
from prefetch_generator import BackgroundGenerator

from datasets.dataset import DocSAM_GT
from models.DocSAM import DocSAM 

from itertools import accumulate
import torch.multiprocessing as mp
from torch.utils.data import Subset



STAGE = "test"
MODEL_SIZE = "base"
SAVE_PATH = './outputs/outputs_test/'

SHORT_RANGE = (704, 896)
PATCH_SIZE = (640, 640)
PATCH_NUM = 1
KEEP_SIZE = False
MAX_NUM = 10

MAX_NUM = 10
BATCH_SIZE = 1
RESTORE_FROM = './snapshots/last_model.pth'
GPU_IDS = '0'


def str2bool(input_str):
    """
    Converts a string input to a boolean value.

    Args:
        input_str (str): A string representation of a boolean value. 
                         Accepted values for True: 'yes', 'true', 't', 'y', '1'.
                         Accepted values for False: 'no', 'false', 'f', 'n', '0'.

    Returns:
        bool: The converted boolean value (True or False).

    Raises:
        argparse.ArgumentTypeError: If the input string does not match any 
                                    of the accepted boolean representations.
    """
    
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_tuple(input_str):
    """
    Parses a string into a tuple of two integers.

    Args:
        input_str (str): A string containing two integers separated by a comma (e.g., '1,2').

    Returns:
        tuple: A tuple of two integers parsed from the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not in the correct format 
                                    or does not contain exactly two integers.
    """
    try:
        parsed_tuple = tuple(map(int, input_str.split(',')))
        if len(parsed_tuple) != 2:
            raise ValueError
        return parsed_tuple
    except ValueError:
        raise argparse.ArgumentTypeError("Input must be two integers separated by a comma (e.g., '1,2')")


def get_arguments():
    """
    Parses command-line arguments for configuring the LPN-ResNet Network.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments with their values.
    """

    parser = argparse.ArgumentParser(description="LPN-ResNet Network")
    
    parser.add_argument('--stage', type=str, default="test", help='Test or inference.')
    parser.add_argument('--model-size', type=str, default=MODEL_SIZE, help='Model size: tiny, small, base, large.')
    parser.add_argument("--eval-path", type=str, nargs='+', help='A list of evaluation paths')
    parser.add_argument("--save-path", type=str, default=SAVE_PATH, help='Path to save outputs')
    parser.add_argument("--short-range", type=parse_tuple, default=SHORT_RANGE, help='Short side range')
    parser.add_argument("--patch-size", type=parse_tuple, default=PATCH_SIZE, help='Patch size sampled from each image during training')
    parser.add_argument("--patch-num", type=int, default=PATCH_NUM, help='Patch number')
    parser.add_argument("--keep-size", type=str2bool, default=KEEP_SIZE, help='Whether to keep original image size (True/False)')
    parser.add_argument('--max-num', type=int, default=MAX_NUM, help='Max image num for evaluation.')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help='Batch size for processing')
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help='Path to restore model from')
    parser.add_argument("--gpus", type=str, default=GPU_IDS, help='Comma-separated GPU IDs')

    return parser.parse_args()


def MakePath(path):
    """
    Creates directory structures if they do not already exist.

    Args:
        path (str): The path to check/create. 

    Returns:
        None
    """

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        return


class CustomSubset(Subset):
    """
    A custom subset class that inherits from torch.utils.data.Subset.

    Args:
        dataset (Dataset): The original dataset.
        indices (list): A list of indices to select from the dataset.

    Attributes:
        collate_fn (callable): The collate function from the parent dataset, if it exists.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.collate_fn = getattr(dataset, "collate_fn", None)
        
        
class DataLoaderX(DataLoader):
    """
    Custom DataLoader that uses a background generator to load data asynchronously.
    """
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_instance_palette(num_cls=2):
    """
    Generates a color palette for visualizing instance segmentation masks.

    Parameters:
    - num_cls: int, number of classes (default=2).

    Returns:
    - palette: list, RGB values for each class.
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (0, 0, 0)        # 0: 'background' 
    for i in range(num_cls * 3 - 3):
        palette[i+3] = random.randint(0, 255)

    return palette


def id_map_to_color(id_map, palette):
    """
    Converts an ID map into a color image based on a given palette.

    Parameters:
    - id_map: ndarray, ID map representing different instances.
    - palette: list, RGB values corresponding to each ID.

    Returns:
    - color_map: ndarray, color image.
    """

    hei, wid = id_map.shape[0], id_map.shape[1]
    color_map = np.zeros((hei, wid, 3), np.uint8)
    for i in range(0, id_map.max()+1):
        color_map[id_map==i, :] = palette[3*i:3*i+3]
    
    return color_map


def bimask_to_id_mask(bimasks):
    """
    Converts multiple binary masks into an instance segmentation mask.

    Parameters:
    - bimasks: Tensor, multiple binary masks.

    Returns:
    - instance_mask: Tensor, resulting instance mask.
    """

    instance_mask = np.zeros(bimasks.shape[1:], np.int32)
    for i, bimask in enumerate(bimasks):
        instance_mask[bimask == True] = i + 1

    return instance_mask


def mask_bbox(masks, img_shape=None, is_norm=False):
    """
    Converts binary masks into bounding boxes.

    Parameters:
    - masks: Tensor, input binary masks.
    - img_shape: tuple, shape of the original image (optional).
    - is_norm: bool, whether to normalize the bounding box coordinates.

    Returns:
    - bboxes: Tensor, resulting bounding boxes.
    """

    Q, H, W = masks.shape
    
    x_projection = masks.any(dim=-2)    # [Q, W]
    y_projection = masks.any(dim=-1)    # [Q, H]
    
    x1 = (x_projection.cumsum(dim=-1) == 1).float().argmax(dim=-1)
    x2 = W - 1 - ((x_projection.flip([-1])).cumsum(dim=-1) == 1).float().argmax(dim=-1)
    
    y1 = (y_projection.cumsum(dim=-1) == 1).float().argmax(dim=-1)
    y2 = H - 1 - ((y_projection.flip([-1])).cumsum(dim=-1) == 1).float().argmax(dim=-1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).float()
        
    if is_norm:
        bboxes[...,[0,2]] = bboxes[...,[0,2]] / W
        bboxes[...,[1,3]] = bboxes[...,[1,3]] / H
    elif img_shape is not None:
        bboxes[...,[0,2]] = bboxes[...,[0,2]] / W * img_shape[-1]
        bboxes[...,[1,3]] = bboxes[...,[1,3]] / H * img_shape[-2]
            
    return bboxes


def mask_iou(masks1, masks2, scale_factor=None):
    """
    Calculates Intersection over Union (IoU) between two sets of masks.

    Parameters:
    - masks1: Tensor, first set of masks.
    - masks2: Tensor, second set of masks.
    - scale_factor: float, optional scaling factor for resizing masks.

    Returns:
    - ious: Tensor, IoU scores.
    """

    if scale_factor is not None:
        masks1 = F.interpolate(masks1[None].float(), scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]
        masks2 = F.interpolate(masks2[None].float(), scale_factor=scale_factor, mode="bilinear", align_corners=False)[0]
        
    masks1 = masks1.float().flatten(1)
    masks2 = masks2.float().flatten(1)
    
    numerator = torch.matmul(masks1, masks2.T)
    denominator = masks1.sum(-1)[:, None] + masks2.sum(-1)[None, :] - numerator
    ious = numerator / denominator.clamp(min=1)
    
    return ious


def mask_iou_group_wise(masks1, masks2, scale_factor=None, group_size1=8, group_size2=8):
    """
    Calculates group-wise Intersection over Union (IoU) between two sets of masks.

    Parameters:
    - masks1: Tensor, first set of masks.
    - masks2: Tensor, second set of masks.
    - scale_factor: float, optional scaling factor for resizing masks.
    - group_size1: int, size of groups for the first set of masks.
    - group_size2: int, size of groups for the second set of masks.

    Returns:
    - ious: Tensor, IoU scores.
    """

    ious = torch.zeros(masks1.size(0), masks2.size(0))
    
    group_num1 = math.ceil(masks1.size(0) / group_size1)
    group_num2 = math.ceil(masks2.size(0) / group_size2)
    for i in range(group_num1):
        start1 = i * group_size1
        end1 = min(start1 + group_size1, masks1.size(0))
        sub_masks1 = masks1[start1:end1,...]
        for j in range(group_num2):
            start2 = j * group_size2
            end2 = min(start2 + group_size2, masks2.size(0))
            sub_masks2 = masks2[start2:end2,...]
            sub_ious = mask_iou(sub_masks1, sub_masks2, scale_factor)
            ious[start1:end1, start2:end2] = sub_ious
            del sub_masks2
        del sub_masks1
    
    return ious


def non_max_suppression(masks, scores, threshold=0.5):  
    """
    Performs Non-Maximum Suppression (NMS) on a set of masks based on their IoU and scores.

    Parameters:
    - masks: Tensor, input binary masks.
    - scores: Tensor, confidence scores for each mask.
    - threshold: float, IoU threshold for NMS.

    Returns:
    - keep_indices: list, indices of masks to keep after NMS.
    """

    # Convert masks to bounding boxes
    bboxes = mask_bbox(masks)
    # Calculate IoU between all pairs of bounding boxes
    bbox_ious = box_iou(bboxes, bboxes).cpu().numpy()
    
    # Sort indices by scores in descending order
    sorted_indices = scores.argsort(descending=True).cpu().numpy()
    order = torch.argsort(scores, descending=True).cpu().numpy()
    keep_indices = []
    
    while np.size(order) > 0:
        # Add the index with the highest score to the keep list
        keep_indices.append(order[0])
        # Calculate IoUs of the current mask with the remaining masks
        ious = bbox_ious[order[0], order[1:]]
        flags = ious > 0.01
        
        if flags.sum() > 0:
            # If there are overlapping masks, calculate mask IoU for more accurate suppression
            mask_ious = mask_iou(masks[order[0]][None,...], masks[order[1:][flags]])
            ious[flags] = mask_ious[0].cpu().numpy()
        
        # Keep only those masks that have an IoU less than the threshold with the current mask
        order = order[1:][ious < threshold]

    return keep_indices


def non_max_suppression_multiclass(masks, bboxes, labels, scores, threshold=0.5):
    """
    Performs Non-Maximum Suppression (NMS) on masks grouped by class labels.

    Parameters:
    - masks: Tensor, input binary masks.
    - bboxes: Tensor, bounding boxes corresponding to the masks.
    - labels: Tensor, class labels for each mask.
    - scores: Tensor, confidence scores for each mask.
    - threshold: float, IoU threshold for NMS.

    Returns:
    - masks: Tensor, filtered masks after NMS.
    - bboxes: Tensor, filtered bounding boxes after NMS.
    - labels: Tensor, filtered labels after NMS.
    - scores: Tensor, filtered scores after NMS.
    """

    # Perform NMS for each unique label separately
    all_keep_indices = []
    for label in labels.unique():
        # Find indices of masks belonging to the current label
        label_indices = [item[0].item() for item in (labels==label).nonzero()]
        # Apply NMS to get indices of masks to keep
        keep_indices = non_max_suppression(masks[label_indices], scores[label_indices], threshold)
        # Map the kept indices back to the original indices
        label_indices = [label_indices[item] for item in keep_indices]
        all_keep_indices += label_indices
    
    return masks[all_keep_indices], bboxes[all_keep_indices], labels[all_keep_indices], scores[all_keep_indices]


def post_process_instance_segmentation(
        pred_instance_masks: torch.Tensor,
        pred_instance_bboxes: torch.Tensor,
        pred_instance_labels: torch.Tensor,
        pred_semantic_masks: torch.Tensor,
        oriimg_size: Tuple[int, int],
        target_size: Tuple[int, int],
        threshold_prob: float = 0.01,
        threshold_nms:  float = 0.25,
    ) -> List[Dict]:
        """
        Processes predicted instance masks, bounding boxes, labels, and semantic masks.
        
        Parameters:
        - pred_instance_masks: Tensor, predicted instance masks.
        - pred_instance_bboxes: Tensor, predicted instance bounding boxes.
        - pred_instance_labels: Tensor, predicted instance labels.
        - pred_semantic_masks: Tensor, predicted semantic masks.
        - oriimg_size: Tuple, original image size.
        - target_size: Tuple, target image size after processing.
        - threshold_prob: float, probability threshold for filtering instances.
        - threshold_nms: float, IoU threshold for non-maximum suppression.
        
        Returns:
        - outputs: List of processed tensors including instance masks, bboxes, labels, scores, semantic masks, labels, and scores.
        """

        # Filter instances based on their confidence scores
        instance_scores, instance_labels = pred_instance_labels.softmax(dim=-1)[:, :-1].max(dim=-1)
        indexes = instance_scores >= threshold_prob
        if indexes.sum() == 0:
            indexes[0] = True
        pred_instance_masks  = pred_instance_masks[indexes]
        pred_instance_bboxes = pred_instance_bboxes[indexes]
        pred_instance_labels = pred_instance_labels[indexes]

        # Further filter based on mask predictions
        indexes = (pred_instance_masks.sigmoid() > 0.5).sum(dim=[1,2]) > 0
        if indexes.sum() == 0:
            indexes[0] = True
        pred_instance_masks  = pred_instance_masks[indexes]
        pred_instance_bboxes = pred_instance_bboxes[indexes]
        pred_instance_labels = pred_instance_labels[indexes]

        # Resize and process instance masks
        mask_logits = F.interpolate(pred_instance_masks[None], size=target_size, mode="bilinear", align_corners=False)[0].sigmoid()
        instance_maskes = mask_logits > 0.5
        mask_scores = [(score * label).sum() / max(label.sum(), 1e-6) for score, label in zip(mask_logits, instance_maskes)]
        mask_scores = torch.stack(mask_scores, dim=0)

        # Calculate final instance scores and labels
        instance_scores, instance_labels = pred_instance_labels.softmax(dim=-1)[:, :-1].max(dim=-1)
        instance_scores = instance_scores * mask_scores
        instance_labels = instance_labels + 1   # 0 for "_background_"

        # Adjust bounding boxes to target size
        pred_instance_bboxes[:, [0,2]] = pred_instance_bboxes[:, [0,2]] / oriimg_size[1] * target_size[1]
        pred_instance_bboxes[:, [1,3]] = pred_instance_bboxes[:, [1,3]] / oriimg_size[0] * target_size[0]
        pred_instance_bboxes[:, 2] = pred_instance_bboxes[:, 2] - pred_instance_bboxes[:, 0]
        pred_instance_bboxes[:, 3] = pred_instance_bboxes[:, 3] - pred_instance_bboxes[:, 1]
        instance_bboxes = pred_instance_bboxes
        
        # Resize and process semantic masks
        mask_logits = F.interpolate(pred_semantic_masks[None], size=target_size, mode="bilinear", align_corners=False)[0].sigmoid()
        semantic_maskes = mask_logits > 0.5
        
        # Apply Non-Maximum Suppression (NMS)
        instance_maskes, instance_bboxes, instance_labels, instance_scores = non_max_suppression_multiclass(\
            instance_maskes, instance_bboxes, instance_labels, instance_scores, threshold=threshold_nms)
        
        outputs = [instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes]
        
        return outputs


def get_instance_segmentation_results(
        seg_results: List[torch.Tensor],
        image_bbox: Tuple[int, int] = None,
        class_names: List[str] = None,
    ) -> List[Dict]:
        """
        Extracts and sorts instance segmentation results.

        Parameters:
        - seg_results: List of Tensors, processed segmentation results.
        - image_bbox: Tuple, bounding box coordinates of the image region.
        - class_names: List of strings, names of classes.

        Returns:
        - results: Dictionary containing instance masks, bounding boxes, scores, labels, semantic masks, scores, and labels.
        """

        instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes = seg_results
        x1, y1, x2, y2 = image_bbox
        instance_maskes = instance_maskes[:,y1:y2,x1:x2]
        semantic_maskes = semantic_maskes[:,y1:y2,x1:x2]
        instance_bboxes[:, 0] -= x1
        instance_bboxes[:, 1] -= y1
        
        # Sort instances by area from largest to smallest
        instances = []
        for j in range(instance_maskes.size(0)):
            instance = {
                "instance_mask":  instance_maskes[j],
                "instance_bbox":  instance_bboxes[j],
                "instance_score": instance_scores[j],
                "instance_label": instance_labels[j],
            }
            instances.append(instance)

        instances  = sorted(instances, key=lambda instances: instances["instance_mask"].sum(), reverse=True)
        
        # Prepare final lists of tensors
        instance_maskes = [item["instance_mask"]  for item in instances]
        instance_bboxes = [item["instance_bbox"]  for item in instances]
        instance_scores = [item["instance_score"] for item in instances]
        instance_labels = [item["instance_label"] for item in instances]
        
        instance_maskes = torch.stack(instance_maskes, dim=0)
        instance_bboxes = torch.stack(instance_bboxes, dim=0)
        instance_scores = torch.stack(instance_scores, dim=0)
        instance_labels = torch.stack(instance_labels, dim=0) 
                
        # Process semantic masks
        semantic_maskes = semantic_maskes[:len(class_names)-1,...]

        results = { "instance_maskes": instance_maskes, 
                    "instance_bboxes": instance_bboxes, 
                    "instance_scores": instance_scores,
                    "instance_labels": instance_labels, 
                    "semantic_maskes": semantic_maskes, 
                }
        
        return results
    

def predict_whole(model, batch, layer_idx=-1):
    """
    Performs whole image prediction using a given model.

    Parameters:
    - model: Model used for prediction. This can be a standalone model or a DistributedDataParallel model.
    - batch: Batch data including pixel values, mask, and class names.
             - pixel_values: Tensor, preprocessed image pixel values.
             - pixel_mask: Tensor, binary mask indicating valid regions in the pixel values.
             - class_names: List of strings, names of classes corresponding to the dataset.
    - layer_idx: Index of the transformer decoder layer to use for prediction. Default is the last layer (-1).

    Returns:
    - batch_results: List of dictionaries containing prediction results for each image in the batch.
                     Each dictionary includes instance masks, bounding boxes, scores, labels, and semantic masks.
    """

    # Extract batch components: pixel values, pixel mask, and class names
    pixel_values, pixel_mask, class_names = batch["pixel_values"], batch["pixel_mask"], batch["class_names"]

    # Perform forward pass through the model to get predictions
    if isinstance(model, nn.parallel.DistributedDataParallel):  
        # If the model is wrapped with DistributedDataParallel, access the underlying module
        pred = model.module.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, class_names=class_names,)
    else:
        # Otherwise, directly call the mask2former method of the model
        pred = model.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, class_names=class_names,)

    # Initialize a list to store prediction results for each image in the batch
    batch_results = []
    
    # Process each image in the batch individually
    for i in range(pixel_values.size(0)):
        # Post-process instance segmentation results for the current image
        seg_results = post_process_instance_segmentation(
            pred_instance_masks=pred.transformer_decoder_instance_masks[layer_idx][i],
            pred_instance_bboxes=pred.transformer_decoder_bbox_predictions[layer_idx][i],
            pred_instance_labels=pred.transformer_decoder_cate_predictions[layer_idx][i],
            pred_semantic_masks=pred.transformer_decoder_semantic_masks[layer_idx][i],
            oriimg_size=pixel_values.shape[-2:],  # Original image size (height, width)
            target_size=pixel_values.shape[-2:],  # Target size for resizing masks and bboxes
        )
        
        # Further process and extract detailed segmentation results
        seg_results = get_instance_segmentation_results(seg_results, image_bbox=batch["image_bboxes"][i], class_names=batch["class_names"][i])
        batch_results.append(seg_results)

    return batch_results
    

def true_positive_num(true_masks, pred_masks, iou_thres=[0.5]):
    """
    Calculates the number of true positives between predicted and ground truth masks.

    Parameters:
    - true_masks: Tensor, ground truth masks.
    - pred_masks: Tensor, predicted masks.
    - iou_thres: List of floats, IoU thresholds to determine true positives.

    Returns:
    - Tuple containing the number of true masks, predicted masks, and true positives for each threshold.
    """

    # Compute pairwise IoU between all true masks and predicted masks using grouped computation
    mask_ious = mask_iou_group_wise(true_masks, pred_masks, group_size1=100, group_size2=100).cpu().numpy()

    # Count the total number of ground truth masks and predicted masks
    true_num = true_masks.shape[0]
    pred_num = pred_masks.shape[0]

    # Initialize a list to store the number of true positives for each IoU threshold
    tp_nums = [0 for _ in range(len(iou_thres))]

    # Solve the optimal assignment problem to match true masks with predicted masks (maximizing IoU)
    rows, cols = linear_sum_assignment(mask_ious, maximize=True)
    ious = mask_ious[rows, cols]

    # Create a list of matched pairs with their corresponding IoU values
    items = [{"row": row, "col": col, "iou": iou} for row, col, iou in zip(rows, cols, ious)]

    # Sort the matched pairs by IoU in descending order
    items = sorted(items, key=lambda item: item["iou"], reverse=True)
    
    # Calculate the number of true positives for each IoU threshold
    for i in range(len(iou_thres)):
        true_positives = [item for item in items if item["iou"] >= iou_thres[i]]
        tp_nums[i] = len(true_positives)

    return true_num, pred_num, tp_nums


def true_positive_num_multiclass(true_masks, true_labels, pred_masks, pred_labels, pred_scores, labels, iou_thres=[0.5]):
    """
    Extends true_positive_num function to handle multiple classes.

    Parameters:
    - true_masks: Tensor, ground truth masks.
    - true_labels: Tensor, labels for ground truth masks.
    - pred_masks: Tensor, predicted masks.
    - pred_labels: Tensor, labels for predicted masks.
    - pred_scores: Tensor, confidence scores for predicted masks.
    - labels: List of integers, class labels to evaluate.
    - iou_thres: List of floats, IoU thresholds to determine true positives.

    Returns:
    - Lists containing the number of true masks, predicted masks, and true positives for each class and threshold.
    """

    # Initialize lists to store results for each class
    true_nums  = []
    pred_nums = []
    tp_nums = []

    # Convert ground truth masks to binary format
    true_masks = true_masks > 0.5

    # Iterate over each class label
    for label in labels:
        # Filter masks and scores for the current class
        true_masks_label  = true_masks[true_labels == label]
        pred_masks_label  = pred_masks[pred_labels == label]
        pred_scores_label = pred_scores[pred_labels == label]

        # Filter predicted masks based on confidence score threshold (e.g., >= 0.5)
        pred_masks_label  = pred_masks_label[pred_scores_label >= 0.5]

        # Calculate true positives for the current class using the base function
        true_num, pred_num, tp_num = true_positive_num(true_masks_label, pred_masks_label, iou_thres)

        # Append results for the current class to the respective lists
        true_nums.append(true_num)
        pred_nums.append(pred_num)
        tp_nums.append(tp_num)
    
    return true_nums, pred_nums, tp_nums


def sliding_window_crop(image, mask, patch_size):
    """
    Perform sliding window cropping on the input image and mask.
    
    Parameters:
        image (torch.Tensor): Input image tensor of shape (C, H, W), where C is the number of channels,
                            H is the height, and W is the width.
        mask (torch.Tensor): Input mask tensor of shape (1, H, W) or (C, H, W). It should have the same
                            spatial dimensions as the image.
        patch_size (tuple): Target size of each cropped patch, specified as (height, width).
    
    Returns:
        pixel_values (torch.Tensor): Tensor containing all cropped patches from the image,
                                    of shape (N, C, patch_height, patch_width), where N is the number of patches.
        pixel_mask (torch.Tensor): Tensor containing all cropped patches from the mask,
                                    of shape (N, 1, patch_height, patch_width).
        patch_bboxes (torch.Tensor): Tensor containing bounding boxes of each patch in the original image,
                                    of shape (N, 4), where each row is [x1, y1, x2, y2].
        image_bboxes (torch.Tensor): Tensor containing bounding boxes of each patch in the local coordinate system,
                                    of shape (N, 4), where each row is [0, 0, patch_width, patch_height].
    """

    # Extract the dimensions of the input image
    C, H, W = image.size()

    # Initialize lists to store cropped patches, masks, and their corresponding bounding boxes
    pixel_values = []
    pixel_mask   = []
    patch_bboxes = []
    image_bboxes = []
    
    # Preprocess the initial data: resize the entire image and mask to the target patch size using interpolation
    # This ensures that the full image is included as a single patch at the beginning
    pixel_values.append(F.interpolate(image[None], size=patch_size, mode="area")[0])  # Resize image using area interpolation
    pixel_mask.append(F.interpolate(mask[None], size=patch_size, mode="nearest")[0])  # Resize mask using nearest-neighbor interpolation
    patch_bboxes.append([0, 0, W, H])  # Bounding box for the full image in the original coordinate system
    image_bboxes.append([0, 0, patch_size[1], patch_size[0]])  # Bounding box for the full image in the local coordinate system
    
    # Calculate the stride for vertical and horizontal directions (half the patch size)
    stride_v, stride_h = patch_size[0] // 2, patch_size[1] // 2

    # Compute the number of patches required in the vertical and horizontal directions
    patch_rows = int(math.ceil((H - patch_size[0]) / stride_v) + 1)  # Number of rows of patches
    patch_cols = int(math.ceil((W - patch_size[1]) / stride_h) + 1)  # Number of columns of patches
    
    # Iterate over all possible patches using the sliding window approach
    for row in range(patch_rows):
        for col in range(patch_cols):
            # Calculate the top-left (y1, x1) and bottom-right (y2, x2) coordinates of the current patch
            y1 = int(row * stride_v)  # Top edge of the patch
            y2 = min(y1 + patch_size[0], H)  # Bottom edge of the patch (clamped to image height)
            y1 = max(y2 - patch_size[0], 0)  # Adjust top edge if necessary to ensure patch size is maintained
            x1 = int(col * stride_h)  # Left edge of the patch
            x2 = min(x1 + patch_size[1], W)  # Right edge of the patch (clamped to image width)
            x1 = max(x2 - patch_size[1], 0)  # Adjust left edge if necessary to ensure patch size is maintained
            
            # Append the cropped patch, mask, and bounding boxes for the current patch
            pixel_values.append(image[:, y1:y2, x1:x2])
            pixel_mask.append(mask[:, y1:y2, x1:x2])
            patch_bboxes.append([x1, y1, x2, y2])
            image_bboxes.append([0, 0, x2 - x1, y2 - y1])
            
    # Stack all cropped patches, masks, and bounding boxes into tensors
    pixel_values = torch.stack(pixel_values, dim=0)
    pixel_mask   = torch.stack(pixel_mask, dim=0)
    patch_bboxes = torch.tensor(patch_bboxes).long()
    image_bboxes = torch.tensor(image_bboxes).long()
    
    return pixel_values, pixel_mask, patch_bboxes, image_bboxes
    

def predict_with_dynamic_batch_size(model, pixel_values, pixel_mask, image_bboxes, class_names, batch_size=64, layer_idx=-1):
    """
    Perform predictions using a dynamic batch size to handle memory constraints.

    Parameters:
        model: The neural network model used for prediction. Can be a standalone model or wrapped with DistributedDataParallel.
        pixel_values (torch.Tensor): Input image patches of shape (N, C, H, W), where N is the number of patches,
                                    C is the number of channels, and H, W are the height and width of each patch.
        pixel_mask (torch.Tensor): Binary mask tensor corresponding to `pixel_values`, of shape (N, 1, H, W).
        image_bboxes (torch.Tensor): Bounding boxes of each patch in the original image, of shape (N, 4).
        class_names (list): List of class names corresponding to the dataset.
        batch_size (int): Initial batch size for prediction. Default is 64.
        layer_idx (int): Index of the transformer decoder layer to use for prediction. Default is -1 (last layer).

    Returns:
        instance_masks (torch.Tensor): Predicted instance masks for all patches, padded to the same size.
        bbox_predictions (torch.Tensor): Predicted bounding boxes for all patches, padded to the same size.
        cate_predictions (torch.Tensor): Predicted category logits for all patches, padded to the same size.
        semantic_masks (torch.Tensor): Predicted semantic masks for all patches, padded to the same size.
    """

    # Total number of patches to process
    patch_num = pixel_values.shape[0]

    # Ensure the batch size does not exceed the number of patches
    batch_size = min(batch_size, pixel_values.shape[0])

    # Dynamically adjust batch size if out-of-memory errors occur
    while batch_size >= 1:
        try:
            # Initialize lists to store predictions for all patches
            instance_masks = []
            bbox_predictions = []
            cate_predictions = []
            semantic_masks = []

            # Calculate the number of batches needed based on the current batch size
            batch_num = math.ceil(patch_num / batch_size)

            # Process each batch sequentially
            for i in range(batch_num):
                print(f"Predicting batch: {i}/{batch_num}, batch size: {batch_size}")
                start = i * batch_size  # Start index of the current batch
                end = min((i + 1) * batch_size, patch_num)  # End index of the current batch

                # Perform forward pass through the model
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    # If the model is wrapped with DistributedDataParallel, access the underlying module
                    batch_preds = model.module.mask2former(
                        pixel_values=pixel_values[start:end, ...],
                        pixel_mask=pixel_mask[start:end, ...],
                        image_bboxes=image_bboxes[start:end, ...],
                        class_names=class_names[start:end],
                    )
                else:
                    # Otherwise, directly call the mask2former method of the model
                    batch_preds = model.mask2former(
                        pixel_values=pixel_values[start:end, ...],
                        pixel_mask=pixel_mask[start:end, ...],
                        image_bboxes=image_bboxes[start:end, ...],
                        class_names=class_names[start:end],
                    )

                # Extract predictions from the specified decoder layer and split them by batch dimension
                instance_masks += [item[0] for item in batch_preds.transformer_decoder_instance_masks[layer_idx].split(1, dim=0)]
                bbox_predictions += [item[0] for item in batch_preds.transformer_decoder_bbox_predictions[layer_idx].split(1, dim=0)]
                cate_predictions += [item[0] for item in batch_preds.transformer_decoder_cate_predictions[layer_idx].split(1, dim=0)]
                semantic_masks += [item[0] for item in batch_preds.transformer_decoder_semantic_masks[layer_idx].split(1, dim=0)]

            # Pad predictions to ensure they have the same size across all patches
            instance_masks = pad_sequence(instance_masks, batch_first=True, padding_value=-1e10)
            bbox_predictions = pad_sequence(bbox_predictions, batch_first=True, padding_value=0)
            cate_predictions = pad_sequence(cate_predictions, batch_first=True, padding_value=0)
            semantic_masks = pad_sequence(semantic_masks, batch_first=True, padding_value=-1e10)

            return instance_masks, bbox_predictions, cate_predictions, semantic_masks

        except torch.cuda.OutOfMemoryError:
            # Handle out-of-memory errors by reducing the batch size
            print(f"Out of memory with batch size: {batch_size}. Reducing batch size.")
            torch.cuda.empty_cache()  # Clear GPU memory cache
            batch_size = math.ceil(batch_size / 2)  # Halve the batch size and retry


def predict_slide_window(model, batch, patch_size, layer_idx=-1):
    """
    Performs prediction using a sliding window technique.

    Parameters:
    - model: Model used for prediction. Can be a standalone model or wrapped with DistributedDataParallel.
    - batch (dict): Dictionary containing the following keys:
        - "pixel_values" (torch.Tensor): Input image tensor of shape (N, C, H, W), where N is the batch size,
                                         C is the number of channels, and H, W are the height and width.
        - "pixel_mask" (torch.Tensor): Binary mask tensor corresponding to `pixel_values`, of shape (N, 1, H, W).
        - "class_names" (list): List of class names for each image in the batch.
        - "image_bboxes" (torch.Tensor): Bounding boxes of the original images, of shape (N, 4).
    - patch_size (tuple): Size of patches for sliding window, specified as (height, width).
    - layer_idx (int): Index of the transformer decoder layer to use for prediction. Default is -1 (last layer).

    Returns:
    - batch_results (list): A list of dictionaries, where each dictionary contains the final prediction results
                            for an image in the batch, including instance masks, bounding boxes, labels, scores,
                            and semantic masks.
    """

    # Initialize lists to store data for all patches across the batch
    all_pixel_values = []
    all_pixel_mask   = []
    all_patch_bboxes = []
    all_image_bboxes = []
    all_class_names  = []
    image_patch_nums = []
    
    # Perform sliding window cropping for each image in the batch
    for image, mask, class_names in zip(batch["pixel_values"], batch["pixel_mask"], batch["class_names"]):
        pixel_values, pixel_mask, patch_bboxes, image_bboxes = sliding_window_crop(image, mask, patch_size)
        all_pixel_values.append(pixel_values)
        all_pixel_mask.append(pixel_mask)
        all_patch_bboxes.append(patch_bboxes)
        all_image_bboxes.append(image_bboxes)
        all_class_names += [class_names] * pixel_values.shape[0]  # Repeat class names for each patch
        image_patch_nums.append(pixel_values.shape[0])  # Store the number of patches for this image
        
    # Concatenate all patches into single tensors for batch processing
    all_pixel_values = torch.cat(all_pixel_values, dim=0)
    all_pixel_mask   = torch.cat(all_pixel_mask, dim=0)
    all_image_bboxes = torch.cat(all_image_bboxes, dim=0)
    
    # Perform predictions using dynamic batch size to handle memory constraints
    all_instance_masks, all_bbox_predictions, all_cate_predictions, all_semantic_masks = \
        predict_with_dynamic_batch_size(model, all_pixel_values, all_pixel_mask, all_image_bboxes, all_class_names, batch_size=16, layer_idx=layer_idx)
    
    # Split predictions back into per-image results based on the number of patches per image
    all_instance_masks   = all_instance_masks.split(image_patch_nums, dim=0)
    all_bbox_predictions = all_bbox_predictions.split(image_patch_nums, dim=0)
    all_cate_predictions = all_cate_predictions.split(image_patch_nums, dim=0)
    all_semantic_masks   = all_semantic_masks.split(image_patch_nums, dim=0)
    
    # Post-processing step to merge overlapping predictions and perform non-max suppression
    batch_results = []
    for n in range(len(batch["pixel_values"])):  # Iterate over each image in the batch
        C, H, W = batch["pixel_values"][n].shape  # Original image dimensions
        
        # Extract predictions for the current image
        instance_masks = all_instance_masks[n]
        bbox_predictions = all_bbox_predictions[n]
        cate_predictions = all_cate_predictions[n]
        semantic_masks = all_semantic_masks[n]
        patch_bboxes = all_patch_bboxes[n]
        
        # Create a count map to track overlapping regions during merging
        count_map = torch.ones(1, H, W).to(semantic_masks.device)
        
        # Merge predictions for each image and apply post-processing steps
        for p in range(instance_masks.shape[0]):  # Iterate over each patch
            target_size = (H, W) if p == 0 else patch_size  # Use full image size for the first patch
            seg_results = post_process_instance_segmentation(
                pred_instance_masks=instance_masks[p],
                pred_instance_bboxes=bbox_predictions[p],
                pred_instance_labels=cate_predictions[p],
                pred_semantic_masks=semantic_masks[p],
                oriimg_size=patch_size,
                target_size=target_size,
            )
            
            # Combine results for whole images and patches
            if p == 0:
                # Apply weighting to the first patch (full image) predictions
                seg_results[3] = seg_results[3] * 0.75
                seg_results[4] = seg_results[4].float()
                batch_results.append(seg_results)
            else:
                # Update and merge instance segmentation results for overlapping patches
                instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes = seg_results
                
                # Detect boundary instances and reduce their confidence scores
                bboxes = mask_bbox(instance_maskes)
                indexes = (bboxes[:, 0] < 4) | (bboxes[:, 1] < 4) | (bboxes[:, 2] > patch_size[1]-4) | (bboxes[:, 3] > patch_size[0]-4)
                instance_scores[indexes] = instance_scores[indexes] * 0.5
                
                # Map patch coordinates back to the original image coordinate system
                x1, y1, x2, y2 = patch_bboxes[p]
                instance_maskes = F.pad(instance_maskes, (x1, W-x2, y1, H-y2), "constant", 0)
                instance_bboxes[:, 0] += x1
                instance_bboxes[:, 1] += y1

                # Merge instance segmentation results
                batch_results[n][0] = torch.cat((batch_results[n][0], instance_maskes), dim=0)
                batch_results[n][1] = torch.cat((batch_results[n][1], instance_bboxes), dim=0)
                batch_results[n][2] = torch.cat((batch_results[n][2], instance_labels), dim=0)
                batch_results[n][3] = torch.cat((batch_results[n][3], instance_scores), dim=0)
                batch_results[n][4][:, y1:y2, x1:x2] += semantic_maskes.float()
                count_map[:, y1:y2, x1:x2] += 1
                
                # Apply Non-Maximum Suppression (NMS) to remove redundant predictions
                batch_results[n][0], batch_results[n][1], batch_results[n][2], batch_results[n][3] = non_max_suppression_multiclass(
                    batch_results[n][0], batch_results[n][1], batch_results[n][2], batch_results[n][3], threshold=0.5)

        # Finalize predictions by averaging overlapping regions and thresholding semantic masks
        batch_results[n][4] = (batch_results[n][4] / count_map) > 0.5
        batch_results[n] = get_instance_segmentation_results(
            batch_results[n], image_bbox=batch["image_bboxes"][n], class_names=batch["class_names"][n]
        )
        
    return batch_results

    
def evaluate(args, model, dataloader, gpu_id=0, save_num=50, stage="test"):
    """
    Evaluates the performance of a model on a given dataset using metrics such as IoU, precision, recall, etc.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., patch size, save path).
    - model: The trained model to be evaluated.
    - dataloader: DataLoader providing batches of data for evaluation.
    - gpu_id (int): GPU ID to use for evaluation. Default is 0.
    - save_num (int): Number of images to save visualizations for during evaluation. Default is 50.
    - stage (str): Specifies the evaluation stage ("train" or "test"). Default is "test".

    Returns:
    - coco_gt (dict): Ground truth annotations in COCO format.
    - coco_dt (list): Predicted detections in COCO format.
    - semantic_true_nums (list): True pixel counts for each semantic class.
    - semantic_pred_nums (list): Predicted pixel counts for each semantic class.
    - semantic_correct_nums (list): Correctly predicted pixel counts for each semantic class.
    - all_true_nums (list): True instance counts for each category.
    - all_pred_nums (list): Predicted instance counts for each category.
    - all_tp_nums (list of lists): True positive counts for each category at different IoU thresholds.
    """

    # Set the device to the specified GPU and move the model to the GPU
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device where the model resides

    # Extract categories from the first batch in the dataloader
    for batch in dataloader:
        categories = batch["coco_datas"][0]["categories"]
        break

    # Initialize COCO ground truth and detection structures
    coco_gt = {"images": [], "annotations": [], "categories": categories}
    coco_dt = []

    # Generate color palettes for instance and semantic segmentation visualization
    instance_palette = get_instance_palette(5000)
    semantic_palette = get_instance_palette(len(coco_gt["categories"]) + 2)

    # Initialize counters for semantic segmentation metrics
    semantic_true_nums = [0 for _ in range(len(coco_gt["categories"]))]
    semantic_pred_nums = [0 for _ in range(len(coco_gt["categories"]))]
    semantic_correct_nums = [0 for _ in range(len(coco_gt["categories"]))]

    # Define labels and IoU thresholds for instance segmentation evaluation
    labels = [i + 1 for i in range(len(categories))]
    iou_thres = np.linspace(0.5, 0.95, 10).tolist()
    print("ious:", iou_thres)

    # Initialize counters for instance segmentation metrics
    all_true_nums = [0 for _ in range(len(categories))]
    all_pred_nums = [0 for _ in range(len(categories))]
    all_tp_nums = [[0 for _ in range(len(iou_thres))] for _ in range(len(categories))]

    img_num = 0  # Counter for processed images
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()  # Clear GPU memory cache

            # Move batch data to the specified GPU
            batch['pixel_values'] = [item.cuda(device) for item in batch['pixel_values']] if isinstance(batch['pixel_values'], list) else batch['pixel_values'].cuda(device)
            batch['pixel_mask'] = [item.cuda(device) for item in batch['pixel_mask']] if isinstance(batch['pixel_mask'], list) else batch['pixel_mask'].cuda(device)
            batch['instance_masks'] = [item.cuda(device) for item in batch['instance_masks']]
            batch['instance_bboxes'] = [item.cuda(device) for item in batch['instance_bboxes']]
            batch['instance_labels'] = [item.cuda(device) for item in batch['instance_labels']]
            batch['semantic_masks'] = [item.cuda(device) for item in batch['semantic_masks']]

            # Perform predictions using either whole-image or sliding window approach
            if stage == "train":
                batch_results = predict_whole(model, batch)
            else:
                batch_results = predict_slide_window(model, batch, patch_size=args.patch_size)

            # Process results for each image in the batch
            for i in range(len(batch_results)):
                print("image_name:", batch["image_names"][i])

                # Extract bounding box coordinates and crop the input data accordingly
                x1, y1, x2, y2 = batch["image_bboxes"][i]
                pixel_values = batch['pixel_values'][i][:, y1:y2, x1:x2]
                pixel_mask = batch['pixel_mask'][i][:, y1:y2, x1:x2]
                instance_masks = batch['instance_masks'][i][:, y1:y2, x1:x2]
                semantic_masks = batch['semantic_masks'][i][:-1, y1:y2, x1:x2]
                instance_bboxes = batch['instance_bboxes'][i]
                instance_bboxes[:, [0, 2]] -= x1
                instance_bboxes[:, [1, 3]] -= y1

                # Extract metadata for the current image
                dataset_names = batch['dataset_names'][i]
                image_names = batch['image_names'][i]
                class_names = batch['class_names'][i]

                results = batch_results[i]  # Prediction results for the current image

                # Save visualizations for the first `save_num` images
                if img_num < save_num:
                    # Save the original image
                    image = pixel_values.permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, ::-1]
                    image = np.ascontiguousarray(image)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + ".jpg"))
                    PILImage.fromarray(image).save(os.path.join(args.save_path, dataset_names, image_names + ".jpg"))

                    # Visualize and save semantic ground truth masks
                    image_semantic_gt = copy.deepcopy(image)
                    semantic_gt = bimask_to_id_mask(semantic_masks.cpu().numpy())
                    color_mask = id_map_to_color(semantic_gt, semantic_palette)
                    image_semantic_gt[semantic_gt > 0] = image_semantic_gt[semantic_gt > 0] // 2 + color_mask[semantic_gt > 0] // 2
                    image_semantic_gt = PILImage.fromarray(image_semantic_gt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_semantic_gt.png"))
                    image_semantic_gt.save(os.path.join(args.save_path, dataset_names, image_names + "_semantic_gt.png"))

                    # Visualize and save predicted semantic masks
                    image_semantic_dt = copy.deepcopy(image)
                    semantic_dt = bimask_to_id_mask(results['semantic_maskes'].cpu().numpy())
                    color_mask = id_map_to_color(semantic_dt, semantic_palette)
                    image_semantic_dt[semantic_dt > 0] = image_semantic_dt[semantic_dt > 0] // 2 + color_mask[semantic_dt > 0] // 2
                    image_semantic_dt = PILImage.fromarray(image_semantic_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_semantic_dt.png"))
                    image_semantic_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_semantic_dt.png"))

                    # Visualize and save instance ground truth masks
                    image_instance_gt = copy.deepcopy(image)
                    instance_gt = bimask_to_id_mask(instance_masks.cpu().numpy())
                    color_mask = id_map_to_color(instance_gt, instance_palette)
                    image_instance_gt[instance_gt > 0] = image_instance_gt[instance_gt > 0] // 2 + color_mask[instance_gt > 0] // 2

                    image_instance_dt = copy.deepcopy(image)
                    instance_dt = bimask_to_id_mask(results['instance_maskes'][results['instance_scores'] > 0.5].cpu().numpy())
                    color_mask = id_map_to_color(instance_dt, instance_palette)
                    image_instance_dt[instance_dt > 0] = image_instance_dt[instance_dt > 0] // 2 + color_mask[instance_dt > 0] // 2

                    # Draw bounding boxes and labels on the instance ground truth image
                    for bbox, label in zip(instance_bboxes, batch['instance_labels'][i]):
                        x1, y1, x2, y2 = bbox
                        category_id = label + 1
                        cv2.rectangle(image_instance_gt, (int(x1), int(y1)), (int(x2), int(y2)), color=semantic_palette[category_id * 3:category_id * 3 + 3], thickness=2)
                        class_score = class_names[category_id - 1]
                        cv2.putText(image_instance_gt, class_score, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id * 3:category_id * 3 + 3], 1)

                    image_instance_gt = PILImage.fromarray(image_instance_gt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_instance_gt.png"))
                    image_instance_gt.save(os.path.join(args.save_path, dataset_names, image_names + "_instance_gt.png"))

                    # Visualize and save predicted category masks
                    image_category_dt = copy.deepcopy(image)

                    category_dt = torch.zeros(results['instance_maskes'].shape[-2:]).long()
                    for j in range(results['instance_maskes'].size(0)):
                        if results['instance_scores'][j] >= 0.5:
                            category_dt[results['instance_maskes'][j] == 1] = results['instance_labels'][j]

                    color_mask = id_map_to_color(category_dt, semantic_palette)
                    image_category_dt[category_dt > 0] = image_category_dt[category_dt > 0] // 2 + color_mask[category_dt > 0] // 2
                    image_category_dt = PILImage.fromarray(image_category_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_category_dt.png"))
                    image_category_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_category_dt.png"))

                # Update semantic segmentation metrics
                for c in range(semantic_masks.size(0)):
                    semantic_true_nums[c] += semantic_masks[c].sum().cpu().item()
                    semantic_pred_nums[c] += results["semantic_maskes"][c].sum().cpu().item()
                    semantic_correct_nums[c] += (semantic_masks[c] * results["semantic_maskes"][c]).sum().cpu().item()

                # Update instance segmentation metrics
                true_nums, pred_nums, tp_nums = true_positive_num_multiclass(
                    instance_masks, batch['instance_labels'][i] + 1,
                    results['instance_maskes'], results['instance_labels'], results['instance_scores'], labels=labels, iou_thres=iou_thres
                )
                all_true_nums = [item1 + item2 for item1, item2 in zip(all_true_nums, true_nums)]
                all_pred_nums = [item1 + item2 for item1, item2 in zip(all_pred_nums, pred_nums)]
                all_tp_nums = [[m + n for m, n in zip(item1, item2)] for item1, item2 in zip(all_tp_nums, tp_nums)]

                # Update COCO ground truth and detection structures
                coco_gt["images"] += batch["coco_datas"][i]["images"]
                coco_gt["annotations"] += batch["coco_datas"][i]["annotations"]

                for j in range(results['instance_maskes'].size(0)):
                    image_id = batch["coco_datas"][i]["images"][0]["id"]
                    category_id = results['instance_labels'][j].item()
                    score = results['instance_scores'][j].item()

                    binary_mask = results['instance_maskes'][j].cpu().numpy().astype(np.uint8)
                    segmentation = mask_utils.encode(np.asfortranarray(binary_mask))
                    segmentation['counts'] = segmentation['counts'].decode('utf-8')
                    bbox = results['instance_bboxes'][j].cpu().numpy().tolist()

                    coco_dt.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "score": score,
                    })
                    
                    if img_num < save_num and score >= 0.5:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                        cv2.rectangle(image_instance_dt, (int(x1), int(y1)), (int(x2), int(y2)), color=semantic_palette[category_id*3:category_id*3+3], thickness=2)
                        class_score = class_names[category_id-1] + "-" + str(round(score, 4))
                        cv2.putText(image_instance_dt, class_score, (int(x1), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id*3:category_id*3+3], 1)

                if img_num < save_num:
                    image_instance_dt = PILImage.fromarray(image_instance_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.png"))
                    image_instance_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.png"))
                
                img_num += 1  # Increment the image counter

    # Restore the model to training mode and clear GPU memory cache
    model.train()
    torch.cuda.empty_cache()

    # Return evaluation results
    return (coco_gt, coco_dt, semantic_true_nums, semantic_pred_nums, semantic_correct_nums, all_true_nums, all_pred_nums, all_tp_nums)
    

def evaluate_parallel(args, model, dataset, save_num=50, stage="test", thres_num=1):
    """
    Evaluates the performance of a model on a given dataset in parallel using multiple processes.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., GPU IDs, patch size, save path).
    - model: The trained model to be evaluated.
    - dataset: The dataset to evaluate the model on.
    - save_num (int): Number of images to save visualizations for during evaluation. Default is 50.
    - stage (str): Specifies the evaluation stage ("train" or "test"). Default is "test".
    - thres_num (int): Number of parallel processes to use. If None, it is set to the minimum of CPU count and dataset size.

    Returns:
    - coco_gt (dict): Ground truth annotations in COCO format.
    - coco_dt (list): Predicted detections in COCO format.
    - semantic_true_nums (list): True pixel counts for each semantic class.
    - semantic_pred_nums (list): Predicted pixel counts for each semantic class.
    - semantic_correct_nums (list): Correctly predicted pixel counts for each semantic class.
    - all_true_nums (list): True instance counts for each category.
    - all_pred_nums (list): Predicted instance counts for each category.
    - all_tp_nums (list of lists): True positive counts for each category at different IoU thresholds.
    """

    # Determine the number of parallel processes to use
    thres_num = min(int(mp.cpu_count()), len(dataset)) if thres_num is None else thres_num

    # Divide the dataset into subsets for each process
    base, remainder = len(dataset) // thres_num, len(dataset) % thres_num
    eachThresImNum = [0] + [base + 1 if i < remainder else base for i in range(thres_num)]
    indices = list(accumulate(eachThresImNum))  # Compute cumulative sums to define subset ranges

    # Divide the number of images to save visualizations for across processes
    base, remainder = save_num // thres_num, save_num % thres_num
    eachThresSaveNum = [base + 1 if i < remainder else base for i in range(thres_num)]

    # Create a multiprocessing pool and evaluate each subset in parallel
    with mp.Pool(thres_num) as mp_pool:
        results = []
        for i in range(thres_num):
            gpu_id = i % len(args.gpus.split(","))  # Assign a GPU to each process
            model_i = copy.deepcopy(model)  # Deep copy the model for each process
            subset = CustomSubset(dataset, range(indices[i], indices[i + 1]))  # Create a subset of the dataset
            loader = DataLoaderX(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=subset.collate_fn)
            # Submit the evaluation task for the subset to the multiprocessing pool
            results.append(mp_pool.apply_async(evaluate, args=(args, model_i, loader, gpu_id, eachThresSaveNum[i], stage)))

        # Collect results from all processes
        results = [p.get() for p in results]

        mp_pool.close()  # Close the multiprocessing pool
        mp_pool.join()   # Wait for all processes to complete

    # Aggregate results from all processes
    coco_gt, coco_dt, semantic_true_nums, semantic_pred_nums, semantic_correct_nums, all_true_nums, all_pred_nums, all_tp_nums = results[0]
    for idx in range(1, thres_num):
        # Combine COCO ground truth and detection structures
        coco_gt["images"] += results[idx][0]["images"]
        coco_gt["annotations"] += results[idx][0]["annotations"]
        coco_dt += results[idx][1]

        # Aggregate semantic segmentation metrics
        semantic_true_nums = [item1 + item2 for item1, item2 in zip(semantic_true_nums, results[idx][2])]
        semantic_pred_nums = [item1 + item2 for item1, item2 in zip(semantic_pred_nums, results[idx][3])]
        semantic_correct_nums = [item1 + item2 for item1, item2 in zip(semantic_correct_nums, results[idx][4])]

        # Aggregate instance segmentation metrics
        all_true_nums = [item1 + item2 for item1, item2 in zip(all_true_nums, results[idx][5])]
        all_pred_nums = [item1 + item2 for item1, item2 in zip(all_pred_nums, results[idx][6])]
        all_tp_nums = [[a + b for a, b in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(all_tp_nums, results[idx][7])]

    # Return the aggregated evaluation results
    return (coco_gt, coco_dt, semantic_true_nums, semantic_pred_nums, semantic_correct_nums,all_true_nums, all_pred_nums, all_tp_nums)


def compute_metrics(results):
    """
    Computes evaluation metrics for object detection, instance segmentation, and semantic segmentation.

    Parameters:
    - results (tuple): A tuple containing the following elements:
        - coco_gt (dict): Ground truth annotations in COCO format.
        - coco_dt (list): Predicted detections in COCO format.
        - semantic_true_nums (list): True pixel counts for each semantic class.
        - semantic_pred_nums (list): Predicted pixel counts for each semantic class.
        - semantic_correct_nums (list): Correctly predicted pixel counts for each semantic class.
        - all_true_nums (list): True instance counts for each category.
        - all_pred_nums (list): Predicted instance counts for each category.
        - all_tp_nums (list of lists): True positive counts for each category at different IoU thresholds.

    Returns:
    - bbox_mAP (float): Mean Average Precision (mAP) for object detection (bounding box).
    - mask_mAP (float): Mean Average Precision (mAP) for instance segmentation (mask).
    - mask_mF1 (float): Mean F1-score for instance segmentation across all classes and IoU thresholds.
    - mIoU (float): Mean Intersection over Union (mIoU) for semantic segmentation.
    """

    # Unpack the results tuple
    coco_gt, coco_dt, semantic_true_nums, semantic_pred_nums, semantic_correct_nums, \
    all_true_nums, all_pred_nums, all_tp_nums = results

    # Initialize metrics to be computed
    bbox_mAP = 0  # Object detection mAP (bounding box)
    mask_mAP = 0  # Instance segmentation mAP (mask)
    mask_mF1 = 0  # Instance segmentation mean F1-score
    mIoU = 0      # Semantic segmentation mIoU

    # Save ground truth and detection results as JSON files for COCO evaluation
    MakePath("./temp/coco_gt.json")  # Ensure the directory exists
    with open("./temp/coco_gt.json", "w", encoding="utf8") as f:
        json.dump(coco_gt, f, indent=4, ensure_ascii=False)  # Write ground truth to file

    MakePath("./temp/coco_dt.json")  # Ensure the directory exists
    with open("./temp/coco_dt.json", "w", encoding="utf8") as f:
        json.dump(coco_dt, f, indent=4, ensure_ascii=False)  # Write predictions to file

    # Evaluate object detection mAP using the COCO API
    try:
        cocoGt = COCO("./temp/coco_gt.json")  # Load ground truth annotations
        cocoDt = cocoGt.loadRes("./temp/coco_dt.json")  # Load detection results

        # Evaluate bounding box mAP
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.maxDets = [40, 400, 4000]  # Set maximum detections per image
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        bbox_mAP = cocoEval.stats[0]  # Extract overall mAP
        print("bbox_mAP:", bbox_mAP)
        print(" ")
    except Exception as e:
        print(f"Error during bbox_mAP computation: {e}")

    # Evaluate instance segmentation mAP using the COCO API
    try:
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.maxDets = [40, 400, 4000]  # Set maximum detections per image
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mask_mAP = cocoEval.stats[0]  # Extract overall mAP
        print("mask_mAP:", mask_mAP)
        print(" ")
    except Exception as e:
        print(f"Error during mask_mAP computation: {e}")

    # Compute instance segmentation mean F1-score
    try:
        class_num = 0  # Counter for valid classes
        for i in range(len(coco_gt["categories"])):  # Iterate over categories
            if all_true_nums[i] == 0 and all_pred_nums[i] == 0:
                continue  # Skip classes with no true or predicted instances

            # Compute precision, recall, and F1-score for each IoU threshold
            P = [item / max(all_pred_nums[i], 1) for item in all_tp_nums[i]]  # Precision
            R = [item / max(all_true_nums[i], 1) for item in all_tp_nums[i]]  # Recall
            F = [2 * item / max(all_true_nums[i] + all_pred_nums[i], 1) for item in all_tp_nums[i]]  # F1-score

            # Print detailed metrics for debugging
            print(coco_gt["categories"][i]["name"], all_true_nums[i], all_pred_nums[i], all_tp_nums[i])
            print("precision:", P)
            print("recall:", R)
            print("f1-score:", F)
            print("mean F1:", sum(F) / len(F))

            # Accumulate mean F1-score for all classes
            mask_mF1 += sum(F) / len(F)
            class_num += 1

        # Compute mean F1-score across all classes
        mask_mF1 /= class_num
        print("Mean F1 of all class at all IoUs:", mask_mF1)
        print(" ")
    except Exception as e:
        print(f"Error during mask_mF1 computation: {e}")

    # Compute semantic segmentation mIoU
    try:
        # Calculate IoU for each class and average them
        IoUs = [c / max(a + b - c, 1) for a, b, c in zip(semantic_true_nums, semantic_pred_nums, semantic_correct_nums) if a + b > 0]
        mIoU = sum(IoUs) / len(IoUs)

        # Print detailed metrics for debugging
        print("semantic_true_nums:", semantic_true_nums)
        print("semantic_pred_nums:", semantic_pred_nums)
        print("semantic_correct_nums:", semantic_correct_nums)
        print("IoUs:", IoUs)
        print("mIoU:", mIoU)
    except Exception as e:
        print(f"Error during mIoU computation: {e}")

    # Return computed metrics
    return bbox_mAP, mask_mAP, mask_mF1, mIoU


def evaluate_all_datasets(args, model, stage="test"):
    """
    Function to test a model on all given datasets.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., evaluation paths, GPU settings).
    - model: The trained model to evaluate.
    - stage (str): Specifies the evaluation stage ("train", "val", or "test"). Default is "test".

    Returns:
    - mean_bbox_mAP (float): Mean Average Precision (mAP) for bounding box detection across all datasets.
    - mean_mask_mAP (float): Mean Average Precision (mAP) for instance segmentation masks across all datasets.
    - mean_mask_mF1 (float): Mean F1-score for instance segmentation across all datasets.
    - mean_mIoU (float): Mean Intersection over Union (mIoU) for semantic segmentation across all datasets.
    """

    datas = []
    bbox_mAPs = []
    mask_mAPs = []
    mask_mF1s = []
    mIoUs = []

    # Iterate over all evaluation data paths provided in args
    for index, data_path in enumerate(args.eval_path):
        print(f"Testing: {index + 1} / {len(args.eval_path)} : {data_path}")

        # Extract data and sub-data identifiers from the path for logging purposes
        segs = data_path.split("/")
        
        data = ""
        subdata = ""
        for s, seg in enumerate(segs):
            if seg in {"Ancient", "Handwritten", "Layout", "SceneText", "Table"}:
                data = segs[s + 1]  # Identify dataset name
            if seg in {"train", "test", "val"}:
                subdata = segs[s - 1]  # Identify subset name
        
        data = f"{data} {subdata}" if subdata != data else data
        datas.append(data)

        # Prepare the test dataset and loader
        test_set = DocSAM_GT([data_path], short_range=args.short_range, patch_size=args.patch_size, patch_num=args.patch_num, keep_size=args.keep_size,stage=stage)
        test_set = CustomSubset(test_set, range(0, min(args.max_num, len(test_set))))

        gpu_num = len(args.gpus.split(",")) # Count the number of GPUs available
        if gpu_num == 1 or stage == "train":
            # Single-GPU evaluation
            test_loader = DataLoaderX(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_set.collate_fn)
            bbox_mAP, mask_mAP, mask_mF1, mIoU = compute_metrics(evaluate(args, model, test_loader, gpu_id=0, save_num=50, stage=stage))
        else:
            # Multi-GPU parallel evaluation
            bbox_mAP, mask_mAP, mask_mF1, mIoU = compute_metrics(evaluate_parallel(args, model, test_set, save_num=50, stage=stage, thres_num=gpu_num))
                
        # Store metrics for the current dataset
        bbox_mAPs.append(bbox_mAP)
        mask_mAPs.append(mask_mAP)
        mask_mF1s.append(mask_mF1)
        mIoUs.append(mIoU)

    # Calculate mean metrics across all datasets tested
    mean_bbox_mAP = sum(bbox_mAPs) / len(bbox_mAPs)
    mean_mask_mAP = sum(mask_mAPs) / len(mask_mAPs)
    mean_mask_mF1 = sum(mask_mF1s) / len(mask_mF1s)
    mean_mIoU = sum(mIoUs) / len(mIoUs)

    # Print the metrics for each dataset as well as the overall means
    print("mAPs and mF1s:\n")
    for data, bbox_mAP, mask_mAP, mask_mF1, mIoU in zip(datas, bbox_mAPs, mask_mAPs, mask_mF1s, mIoUs):
        print(f"{data}: bbox_mAP: {bbox_mAP}, mask_mAP: {mask_mAP}, mask_mF1: {mask_mF1}, mIoU: {mIoU}")
    print(f"mean_bbox_mAP: {mean_bbox_mAP}, mean_mask_mAP: {mean_mask_mAP}, mean_mask_mF1: {mean_mask_mF1}, mean_mIoU: {mean_mIoU}")

    return mean_bbox_mAP, mean_mask_mAP, mean_mask_mF1, mean_mIoU


def inference(args, model, dataloader, gpu_id=0, save_num=50, stage="inference"):
    """
    Function to perform inference using a trained model on a dataset.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., save paths, patch size).
    - model (torch.nn.Module): The trained model used for inference.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset for inference.
    - gpu_id (int): ID of the GPU to use for inference. Default is 0.
    - save_num (int): Number of images to visualize and save results for. Default is 50.
    - stage (str): Specifies the stage of inference ("inference" by default).

    Returns:
    - None: This function performs inference and saves results but does not return any value.
    """

    # Set the device to the specified GPU and move the model to the GPU
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    device = next(model.parameters()).device  # Get the device where the model resides

    img_num = 0  # Counter for processed images
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in tqdm(dataloader):  # Iterate over batches from the dataloader
            torch.cuda.empty_cache()  # Clear GPU memory cache to free up space

            # Move batch data to the specified GPU
            batch['pixel_values'] = [item.cuda(device) for item in batch['pixel_values']] if isinstance(batch['pixel_values'], list) else batch['pixel_values'].cuda(device)
            batch['pixel_mask'] = [item.cuda(device) for item in batch['pixel_mask']] if isinstance(batch['pixel_mask'], list) else batch['pixel_mask'].cuda(device)

            # Perform inference using a sliding window approach
            batch_results = predict_slide_window(model, batch, patch_size=args.patch_size)

            # Process results for each image in the batch
            for i in range(len(batch_results)):
                print("image_name:", batch["image_names"][i])

                # Extract bounding box coordinates and crop the input data accordingly
                x1, y1, x2, y2 = batch["image_bboxes"][i]
                pixel_values = batch['pixel_values'][i][:, y1:y2, x1:x2]
                pixel_mask = batch['pixel_mask'][i][:, y1:y2, x1:x2]

                # Extract metadata for the current image
                dataset_names = batch['dataset_names'][i]
                image_names = batch['image_names'][i]
                class_names = batch['class_names'][i]

                results = batch_results[i]  # Prediction results for the current image

                dt = []  # List to store detection results
                # Process and append detection results
                for j in range(results['instance_maskes'].size(0)):
                    image_id = img_num
                    category_id = results['instance_labels'][j].item()
                    category_id = class_names[category_id - 1]  # Map label index to class name
                    score = results['instance_scores'][j].item()

                    # Use mask_utils.encode() to transform mask into RLE format
                    binary_mask = results['instance_maskes'][j].cpu().numpy().astype(np.uint8)
                    segmentation = mask_utils.encode(np.asfortranarray(binary_mask))
                    segmentation['counts'] = segmentation['counts'].decode('utf-8')  # Decode RLE counts
                    bbox = results['instance_bboxes'][j].cpu().numpy().tolist()  # Extract bounding box coordinates

                    dt.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "score": score,
                    })

                # Save detection results to a JSONL file
                MakePath(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.jsonl"))
                with open(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.jsonl"), "w", encoding="utf-8") as f:
                    for item in dt:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                # Define color palettes for visualization
                instance_palette = get_instance_palette(5000)
                semantic_palette = get_instance_palette(len(class_names) + 2)

                # Save visualizations for the first `save_num` images
                if img_num < save_num:
                    # Save the original image
                    image = pixel_values.permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, ::-1]  # Convert to HWC and BGR format
                    image = np.ascontiguousarray(image)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + ".jpg"))
                    PILImage.fromarray(image).save(os.path.join(args.save_path, dataset_names, image_names + ".jpg"))

                    # Visualize and save predicted semantic masks
                    image_semantic_dt = copy.deepcopy(image)
                    semantic_dt = bimask_to_id_mask(results['semantic_maskes'].cpu().numpy())  # Convert binary mask to ID mask
                    color_mask = id_map_to_color(semantic_dt, semantic_palette)  # Apply color palette
                    image_semantic_dt[semantic_dt > 0] = image_semantic_dt[semantic_dt > 0] // 2 + color_mask[semantic_dt > 0] // 2
                    image_semantic_dt = PILImage.fromarray(image_semantic_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_semantic_dt.png"))
                    image_semantic_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_semantic_dt.png"))

                    # Visualize and save instance ground truth masks
                    image_instance_dt = copy.deepcopy(image)
                    instance_dt = bimask_to_id_mask(results['instance_maskes'][results['instance_scores'] > 0.5].cpu().numpy())
                    color_mask = id_map_to_color(instance_dt, instance_palette)
                    image_instance_dt[instance_dt > 0] = image_instance_dt[instance_dt > 0] // 2 + color_mask[instance_dt > 0] // 2

                    # Draw bounding boxes and class labels for high-confidence predictions
                    for j in range(results['instance_maskes'].size(0)):
                        category_id = results['instance_labels'][j].item()
                        score = results['instance_scores'][j].item()
                        bbox = results['instance_bboxes'][j].cpu().numpy().tolist()
                        if score >= 0.5:
                            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                            cv2.rectangle(image_instance_dt, (int(x1), int(y1)), (int(x2), int(y2)), color=semantic_palette[category_id * 3:category_id * 3 + 3], thickness=2)
                            class_score = class_names[category_id - 1] + "-" + str(round(score, 4))
                            cv2.putText(image_instance_dt, class_score, (int(x1), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id * 3:category_id * 3 + 3], 1)

                    image_instance_dt = PILImage.fromarray(image_instance_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.png"))
                    image_instance_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_instance_dt.png"))

                    # Visualize and save predicted category masks
                    image_category_dt = copy.deepcopy(image)

                    category_dt = torch.zeros(results['instance_maskes'].shape[-2:]).long()
                    for j in range(results['instance_maskes'].size(0)):
                        if results['instance_scores'][j] >= 0.5:
                            category_dt[results['instance_maskes'][j] == 1] = results['instance_labels'][j]

                    color_mask = id_map_to_color(category_dt, semantic_palette)
                    image_category_dt[category_dt > 0] = image_category_dt[category_dt > 0] // 2 + color_mask[category_dt > 0] // 2
                    image_category_dt = PILImage.fromarray(image_category_dt)
                    MakePath(os.path.join(args.save_path, dataset_names, image_names + "_category_dt.png"))
                    image_category_dt.save(os.path.join(args.save_path, dataset_names, image_names + "_category_dt.png"))

                img_num += 1  # Increment the image counter

    # Restore the model to training mode and clear GPU memory cache
    model.train()
    torch.cuda.empty_cache()
    
    
def inference_parallel(args, model, dataset, save_num=50, stage="inference", thres_num=None):
    """
    Function to perform parallel inference using multiple processes and GPUs.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., GPU IDs, save paths).
    - model (torch.nn.Module): The trained model used for inference.
    - dataset (torch.utils.data.Dataset): Dataset for inference.
    - save_num (int): Number of images to visualize and save results for. Default is 50.
    - stage (str): Specifies the stage of inference ("inference" by default).
    - thres_num (int): Number of parallel processes to use. If None, it defaults to the minimum of CPU count and dataset size.

    Returns:
    - None: This function performs parallel inference and saves results but does not return any value.
    """

    # Determine the number of parallel processes to use
    thres_num = min(int(mp.cpu_count()), len(dataset)) if thres_num is None else thres_num

    # Divide the dataset into subsets for each process
    base, remainder = len(dataset) // thres_num, len(dataset) % thres_num
    eachThresImNum = [0] + [base + 1 if i < remainder else base for i in range(thres_num)]
    indices = list(accumulate(eachThresImNum))  # Compute cumulative sums to define subset ranges

    # Divide the number of images to save visualizations for across processes
    base, remainder = save_num // thres_num, save_num % thres_num
    eachThresSaveNum = [base + 1 if i < remainder else base for i in range(thres_num)]

    # Create a multiprocessing pool and evaluate each subset in parallel
    with mp.Pool(thres_num) as mp_pool:
        results = []
        for i in range(thres_num):
            gpu_id = i % len(args.gpus.split(","))  # Assign a GPU to each process based on available GPUs
            model_i = copy.deepcopy(model)  # Deep copy the model for each process to avoid conflicts
            subset = CustomSubset(dataset, range(indices[i], indices[i + 1]))  # Create a subset of the dataset for the current process
            loader = DataLoaderX(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=subset.collate_fn)
            
            # Submit the evaluation task for the subset to the multiprocessing pool
            results.append(mp_pool.apply_async(
                inference, 
                args=(args, model_i, loader, gpu_id, eachThresSaveNum[i], stage)
            ))

        # Collect results from all processes
        results = [p.get() for p in results]

        mp_pool.close()  # Close the multiprocessing pool to prevent new tasks from being submitted
        mp_pool.join()   # Wait for all processes to complete before proceeding

    return


def inference_all_datasets(args, model, stage="inference"):
    """
    Function to perform inference on all datasets specified in the evaluation paths.

    Parameters:
    - args: Namespace or object containing configuration parameters (e.g., eval_path, GPUs, patch size).
    - model (torch.nn.Module): The trained model used for inference.
    - stage (str): Specifies the stage of inference ("inference" by default).

    Returns:
    - None: This function performs inference on all datasets and saves results but does not return any value.
    """

    # Iterate over all inference data paths provided in args
    for index, data_path in enumerate(args.eval_path):
        print(f"Testing: {index + 1} / {len(args.eval_path)} : {data_path}")

        # Prepare the test dataset and loader
        test_set = DocSAM_GT([data_path], short_range=args.short_range, patch_size=args.patch_size, patch_num=args.patch_num, keep_size=args.keep_size, stage=stage)
        # Limit the dataset size to `args.max_num` if specified
        test_set = CustomSubset(test_set, range(0, min(args.max_num, len(test_set))))
        
        gpu_num = len(args.gpus.split(","))  # Count the number of GPUs available

        if gpu_num == 1:
            # Single-GPU inference
            test_loader = DataLoaderX(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_set.collate_fn)
            inference(args, model, test_loader, gpu_id=0, save_num=50, stage=stage)
        else:
            # Multi-GPU parallel inference
            inference_parallel(args, model, test_set, save_num=50, stage=stage, thres_num=gpu_num)
            
    return


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        float: Number of trainable parameters in millions.
    """
    
    # Sum up the number of elements in all trainable parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return params / 1e6  # Convert to millions


def load_para_weights(model, restore_from):
    """
    Loads pretrained weights into the model.

    Args:
        model (torch.nn.Module): PyTorch model to load weights into.
        restore_from (str): Path to the pretrained model file.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    
    # Load the pretrained weights from the specified file
    pre_dict = torch.load(restore_from, weights_only=True, map_location=torch.device('cpu'))
    cur_dict = model.state_dict()  # Get the current model's state dictionary
    
    matched_dict = {}  # Dictionary to store matched keys and weights
    unmatched_keys = []  # List to store unmatched keys
    for k in cur_dict.keys():
        if k in pre_dict and cur_dict[k].size() == pre_dict[k].size():
            matched_dict[k] = pre_dict[k]  # Matched key and weight
        else:
            unmatched_keys.append(k)  # Unmatched key
    
    # Log unmatched keys if any
    if unmatched_keys:
        print("Unmatched keys in current model:", unmatched_keys)
    
    # Load matched weights into the model
    model.load_state_dict(matched_dict, strict=False)
    print("Pretrained model loaded!!!", restore_from)
    
    return model


if __name__ == '__main__':
    start = timeit.default_timer()
    
    mp.set_start_method('spawn', force=True)
    
    args = get_arguments()

    if args.gpus != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    
    # Instantiate the DocSAM model
    model = DocSAM(model_size=args.model_size)
    print("total paras:", count_parameters(model))
    
    # Load pretrained weights into the model if a valid path is provided in 'args.restore_from'
    if os.path.isfile(args.restore_from):
        model = load_para_weights(model, args.restore_from)
    #model.cuda()
    
    # Evaluate the model using the test() function. It takes the model, arguments, maximum number of images to process, and the stage.
    print(args.eval_path)
    if args.stage == "train" or args.stage == "test":
        mean_bbox_mAP, mean_mask_mAP, mean_mask_mF1, mean_mIoU = evaluate_all_datasets(args, model, stage=args.stage)
        print("mean_bbox_mAP:", mean_bbox_mAP, "mean_mask_mAP:", mean_mask_mAP, "mean_mask_mF1:", mean_mask_mF1, "mean_mIoU:", mean_mIoU)
    else:
        inference_all_datasets(args, model, stage=args.stage)

    # Record the end time and calculate the total execution time.
    end = timeit.default_timer()
    print('total time:', end-start,'seconds')
