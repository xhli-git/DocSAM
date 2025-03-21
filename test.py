
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
    Converts string input to boolean.

    Args:
        input_str (str): String representation of a boolean value.

    Returns:
        bool: Converted boolean value.
    """
    
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_tuple(input_str):
    try:
        parsed_tuple = tuple(map(int, input_str.split(',')))
        if len(parsed_tuple) != 2:
            raise ValueError
        return parsed_tuple
    except ValueError:
        raise argparse.ArgumentTypeError("Input must be two integers separated by a comma (e.g., '1,2')")


def get_arguments():
    """
    Parses command-line arguments for the LPN-ResNet Network.
    
    Returns:
    - parser.parse_args(): parsed arguments object containing all the specified or default values.
    """

    parser = argparse.ArgumentParser(description="LPN-ResNet Network")
    parser.add_argument('--stage', type=str, default="test", help='Test or inference.') 
    parser.add_argument('--model-size', type=str, default=MODEL_SIZE, help='Model size: tiny, small, base, large.') 
    parser.add_argument("--eval-path", type=str, nargs='+', help='A list of evaluation paths')
    parser.add_argument("--save-path", type=str, default=SAVE_PATH, help='Path to save outputs')
    
    parser.add_argument("--short-range", type=parse_tuple, default=SHORT_RANGE, help='Short side range')
    parser.add_argument("--patch-size", type=parse_tuple, default=PATCH_SIZE, help='Patch size sampled from each image during training')
    parser.add_argument("--patch-num", type=int, default=PATCH_NUM, help='Patch number')
    parser.add_argument("--keep-size", type=str2bool, default=KEEP_SIZE, help='Whether to keep original image size')
    parser.add_argument('--max-num', type=int, default=MAX_NUM, help='Max image num for evaluation.') 

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help='Batch size for processing')
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help='Path to restore model from')
    parser.add_argument("--gpus", type=str, default=GPU_IDS, help='Comma-separated GPU IDs')

    return parser.parse_args()


def MakePath(path):
    """
    Creates directory structures if they do not already exist.

    Parameters:
    - path: str, path to check/create.
    """

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        return


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


def bimask_to_instance_mask(bimasks):
    """
    Converts multiple binary masks into an instance segmentation mask.

    Parameters:
    - bimasks: Tensor, multiple binary masks.

    Returns:
    - instance_mask: Tensor, resulting instance mask.
    """

    instance_mask = torch.zeros(bimasks.size()[1:]).long()
    for i, bimask in enumerate(bimasks):
        instance_mask[bimask > 0.5] = i + 1

    return instance_mask


def bimask_to_semantic_mask(bimasks, labels):
    """
    Converts multiple binary masks into a semantic segmentation mask.

    Parameters:
    - bimasks: Tensor, multiple binary masks.
    - labels: list, labels for each binary mask.

    Returns:
    - semantic_mask: Tensor, resulting semantic mask.
    """

    semantic_mask = torch.zeros(bimasks.shape[1:]).long()
    for i, (bimask, label) in enumerate(zip(bimasks, labels)):
        semantic_mask[bimask > 0.5] = label

    return semantic_mask


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
    
    x_projection = masks.sum(dim=-2) > 0.5  # [N, Q, W]
    y_projection = masks.sum(dim=-1) > 0.5  # [N, Q, H]

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

    # start = timeit.default_timer()
    # print("before nms num:", masks.size())
    
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

    # end = timeit.default_timer()
    # print("after nms num:", len(all_keep_indices))
    # print('nms total time2:', end-start,'seconds')
    
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
        mask_scores = [(score * label).sum() / max(label.sum(), 1e-6) for score, label in zip(mask_logits, semantic_maskes)]
        mask_scores = torch.stack(mask_scores, dim=0)
        semantic_scores = mask_scores
        semantic_labels = torch.arange(semantic_scores.size(0)) + 1 # 0 for "_background_"

        # Apply Non-Maximum Suppression (NMS)
        instance_maskes, instance_bboxes, instance_labels, instance_scores = non_max_suppression_multiclass(\
            instance_maskes, instance_bboxes, instance_labels, instance_scores, threshold=threshold_nms)
        
        outputs = [instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes, semantic_labels, semantic_scores]
        
        return outputs


def get_instance_segmentation_results(
        seg_results: List[torch.Tensor],
        image_bbox: Tuple[int, int] = None,
        class_names: List[str] = None,
        threshold_prob: float = 0.5,
    ) -> List[Dict]:
        """
        Extracts and sorts instance segmentation results.

        Parameters:
        - seg_results: List of Tensors, processed segmentation results.
        - image_bbox: Tuple, bounding box coordinates of the image region.
        - class_names: List of strings, names of classes.
        - threshold_prob: float, probability threshold for filtering instances.

        Returns:
        - results: Dictionary containing instance masks, bounding boxes, scores, labels, semantic masks, scores, and labels.
        """

        instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes, semantic_labels, semantic_scores = seg_results
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
        
        # Generate ID maps for instances and categories
        instance_mask_ids = torch.zeros(instance_maskes.shape[-2:]).long()
        category_mask_ids = torch.zeros(instance_maskes.shape[-2:]).long()

        current_segment_id = 1
        for j in range(instance_maskes.size(0)):
            if not torch.all(instance_maskes[j] == 0) and instance_scores[j] >= threshold_prob:
                instance_mask_ids[instance_maskes[j] == 1] = current_segment_id
                category_mask_ids[instance_maskes[j] == 1] = instance_labels[j]
                current_segment_id += 1
                
        # Process semantic masks
        valid_num = len(class_names) - 1
        semantic_maskes = semantic_maskes[:valid_num,...]
        semantic_scores = semantic_scores[:valid_num,...]
        semantic_labels = semantic_labels[:valid_num,...]
        semantic_mask_ids = bimask_to_semantic_mask(semantic_maskes, semantic_labels)

        results = {"instance_maskes": instance_maskes, 
                    "instance_bboxes": instance_bboxes, 
                    "instance_scores": instance_scores,
                    "instance_labels": instance_labels, 
                    "semantic_maskes": semantic_maskes, 
                    "semantic_scores": semantic_scores, 
                    "semantic_labels": semantic_labels, 
                    "instance_mask_ids": instance_mask_ids, 
                    "category_mask_ids": category_mask_ids, 
                    "semantic_mask_ids": semantic_mask_ids, 
                }
        
        return results
    

def predict_whole(model, batch, layer_idx=-1):
    """
    Performs whole image prediction using a given model.

    Parameters:
    - model: Model used for prediction.
    - batch: Batch data including pixel values, mask, and class names.
    - layer_idx: Index of the transformer decoder layer to use for prediction.

    Returns:
    - batch_results: List of dictionaries containing prediction results for each image in the batch.
    """

    pixel_values, pixel_mask, class_names = batch["pixel_values"], batch["pixel_mask"], batch["class_names"]

    # pred = model(batch)
    if isinstance(model, nn.parallel.DistributedDataParallel):  
        pred = model.module.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, class_names=class_names,)
    else:
        pred = model.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, class_names=class_names,)

    batch_results = []
    for i in range(pixel_values.size(0)):
        seg_results = post_process_instance_segmentation(pred_instance_masks = pred.transformer_decoder_instance_masks[layer_idx][i], 
                                                    pred_instance_bboxes = pred.transformer_decoder_bbox_predictions[layer_idx][i], 
                                                    pred_instance_labels = pred.transformer_decoder_cate_predictions[layer_idx][i], 
                                                    pred_semantic_masks = pred.transformer_decoder_semantic_masks[layer_idx][i],
                                                    oriimg_size = pixel_values.shape[-2:],
                                                    target_size = pixel_values.shape[-2:],)
        seg_results = get_instance_segmentation_results(seg_results, image_bbox=batch["img_bboxes"][i], class_names=batch["class_names"][i])
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

    mask_ious = mask_iou_group_wise(true_masks, pred_masks, group_size1=100, group_size2=100).cpu().numpy()

    true_num = true_masks.shape[0]
    pred_num = pred_masks.shape[0]
    tp_nums = [0 for _ in range(len(iou_thres))]

    rows, cols = linear_sum_assignment(mask_ious, maximize=True)
    ious = mask_ious[rows, cols]
    items = [{"row": row, "col": col, "iou": iou,} for row, col, iou in zip(rows, cols, ious)]
    items = sorted(items, key=lambda item: item["iou"], reverse=True)
    
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

    true_nums  = []
    pred_nums = []
    tp_nums = []

    true_masks = true_masks > 0.5
    for label in labels:
        true_masks_label  = true_masks[true_labels==label]
        pred_masks_label  = pred_masks[pred_labels==label]
        pred_scores_label = pred_scores[pred_labels==label]
        pred_masks_label  = pred_masks_label[pred_scores_label >= 0.5]

        true_num, pred_num, tp_num = true_positive_num(true_masks_label, pred_masks_label, iou_thres)

        true_nums.append(true_num)
        pred_nums.append(pred_num)
        tp_nums.append(tp_num)
    
    return true_nums, pred_nums, tp_nums


def predict_slide_window(model, batch, patch_size, layer_idx=-1):
    """
    Performs prediction using a sliding window technique.

    Parameters:
    - model: Model used for prediction.
    - batch: Dictionary containing pixel values, masks, bounding boxes, and class names.
    - patch_size: Tuple, size of patches for sliding window.
    - layer_idx: Integer, index of the transformer decoder layer to use for prediction.

    Returns:
    - List of dictionaries containing prediction results for each image in the batch.
    """

    N, C, H, W = batch["pixel_values"].size()

    pixel_values = []
    pixel_mask   = []
    img_bboxes   = []
    patch_bboxes = []
    
    # Prepare initial data for the sliding window by interpolating the original images/masks to the patch size
    pixel_values.append(F.interpolate(batch["pixel_values"], size=patch_size, mode="area"))
    pixel_mask.append(F.interpolate(batch["pixel_mask"], size=patch_size, mode="nearest"))
    patch_bboxes.append([0, 0, patch_size[1], patch_size[0]])
    
    # Scale the image bounding boxes according to the new size
    scale_y = patch_size[0] / batch["pixel_values"].shape[2]
    scale_x = patch_size[1] / batch["pixel_values"].shape[3]
    img_bboxes_resized = batch["img_bboxes"].clone().detach()
    img_bboxes_resized[:,[0,2]] = (img_bboxes_resized[:,[0,2]] * scale_x).long()
    img_bboxes_resized[:,[1,3]] = (img_bboxes_resized[:,[1,3]] * scale_y).long()
    img_bboxes.append(img_bboxes_resized)
    
    # Calculate stride and number of patches needed for vertical and horizontal directions
    stride_v, stride_h = patch_size[0]//2, patch_size[1]//2
    patch_rows = int(math.ceil((H - patch_size[0]) / stride_v) + 1) 
    patch_cols = int(math.ceil((W - patch_size[1]) / stride_h) + 1) 
    print(f"\nNeed {patch_rows} x {patch_cols} prediction patches")
    
     # Iterate over all patches
    for row in range(patch_rows):
        for col in range(patch_cols):
            y1 = int(row * stride_v)
            y2 = min(y1 + patch_size[0], H)
            y1 = max(y2 - patch_size[0], 0)
            x1 = int(col * stride_h)
            x2 = min(x1 + patch_size[1], W)
            x1 = max(x2 - patch_size[1], 0)
            
            # Append cropped pixel values, masks, and updated bounding boxes for each patch
            pixel_values.append(batch["pixel_values"][:,:,y1:y2,x1:x2])
            pixel_mask.append(batch["pixel_mask"][:,:,y1:y2,x1:x2])
            patch_bboxes.append([x1, y1, x2, y2])
            img_bboxes_cropped = batch["img_bboxes"].clone().detach()
            img_bboxes_cropped[:,[0,2]] = (img_bboxes_cropped[:,[0,2]] - x1).clamp(min=0, max=patch_size[1])
            img_bboxes_cropped[:,[1,3]] = (img_bboxes_cropped[:,[1,3]] - y1).clamp(min=0, max=patch_size[0])
            img_bboxes.append(img_bboxes_cropped)
            
            
    print("Total patch num:", len(patch_bboxes))
    
    # Concatenate all prepared data into tensors
    pixel_values = torch.cat(pixel_values, dim=0)
    pixel_mask = torch.cat(pixel_mask, dim=0)
    img_bboxes = torch.cat(img_bboxes, dim=0)
    
    # Prediction process
    batch_size = 7
    if len(patch_bboxes) <= batch_size:
        # Perform prediction on the entire batch if it fits within the specified batch size
        if isinstance(model, nn.parallel.DistributedDataParallel):
            preds = model.module.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, img_bboxes=img_bboxes, class_names=batch["class_names"] * len(patch_bboxes),)
        else:
            preds = model.mask2former(pixel_values=pixel_values, pixel_mask=pixel_mask, img_bboxes=img_bboxes, class_names=batch["class_names"] * len(patch_bboxes),)
    else:
        # Handle larger batches by splitting them into smaller chunks
        preds = None
        instance_masks = []
        bbox_predictions = []
        cate_predictions = []
        semantic_masks = []
        
        # Process each chunk separately and aggregate results
        for i in range(math.ceil(len(patch_bboxes) / batch_size)):
            print("batch:", i)
            start = i * batch_size
            end = min((i + 1) * batch_size, len(patch_bboxes))

            if isinstance(model, nn.parallel.DistributedDataParallel):
                batch_preds = model.module.mask2former(pixel_values=pixel_values[start*N:end*N,...], pixel_mask=pixel_mask[start*N:end*N,...], \
                                                       img_bboxes=img_bboxes[start*N:end*N,...], class_names=batch["class_names"] * (end - start),)
            else:
                batch_preds = model.mask2former(pixel_values=pixel_values[start*N:end*N,...], pixel_mask=pixel_mask[start*N:end*N,...], \
                    img_bboxes=img_bboxes[start*N:end*N,...], class_names=batch["class_names"] * (end - start),)
            
            # Aggregate predictions
            if preds is None:
                preds = batch_preds
                
            instance_masks   += [item[0] for item in batch_preds.transformer_decoder_instance_masks[layer_idx].split(1, dim=0)]
            bbox_predictions += [item[0] for item in batch_preds.transformer_decoder_bbox_predictions[layer_idx].split(1, dim=0)]
            cate_predictions += [item[0] for item in batch_preds.transformer_decoder_cate_predictions[layer_idx].split(1, dim=0)]
            semantic_masks   += [item[0] for item in batch_preds.transformer_decoder_semantic_masks[layer_idx].split(1, dim=0)]

        preds.transformer_decoder_instance_masks[layer_idx] = pad_sequence(instance_masks, batch_first=True, padding_value=-1e10)
        preds.transformer_decoder_bbox_predictions[layer_idx] = pad_sequence(bbox_predictions, batch_first=True, padding_value=0)
        preds.transformer_decoder_cate_predictions[layer_idx] = pad_sequence(cate_predictions, batch_first=True, padding_value=0)
        preds.transformer_decoder_semantic_masks[layer_idx] = pad_sequence(semantic_masks, batch_first=True, padding_value=-1e10)

    #print("Inference done!")

    # Post-processing step to merge overlapping predictions and perform non-max suppression
    batch_results = []
    count_map = torch.ones(1, H, W).to(pixel_values.device)
    
    # Merge predictions for each image and apply post-processing steps
    for i in range(len(patch_bboxes) * N):
        target_size = (H, W) if i < N else patch_size
        seg_results = post_process_instance_segmentation(pred_instance_masks = preds.transformer_decoder_instance_masks[layer_idx][i], 
                                                    pred_instance_bboxes = preds.transformer_decoder_bbox_predictions[layer_idx][i], 
                                                    pred_instance_labels = preds.transformer_decoder_cate_predictions[layer_idx][i], 
                                                    pred_semantic_masks = preds.transformer_decoder_semantic_masks[layer_idx][i], 
                                                    oriimg_size = patch_size,
                                                    target_size = target_size,)
        
        # Combine results for whole images and patches
        if i < N:
            seg_results[3] = seg_results[3] * 0.75
            seg_results[4] = seg_results[4].float()
            batch_results.append(seg_results)
        else:
            # Update and merge instance segmentation results for patches
            instance_maskes, instance_bboxes, instance_labels, instance_scores, semantic_maskes, semantic_labels, semantic_scores = seg_results
    
            bboxes = mask_bbox(instance_maskes)
            indexes = (bboxes[:, 0] < 4) | (bboxes[:, 1] < 4) | (bboxes[:, 2] > x2-x1-4) | (bboxes[:, 3] > y2-y1-4)
            instance_scores[indexes] = instance_scores[indexes] * 0.5
            
            x1, y1, x2, y2 = patch_bboxes[i // N]
            instance_maskes = F.pad(instance_maskes, (x1, W-x2, y1, H-y2), "constant", 0)
            instance_bboxes[:, 0] += x1
            instance_bboxes[:, 1] += y1

            #print("patch", instance_maskes.size())
            batch_results[i%N][0] = torch.cat((batch_results[i%N][0], instance_maskes), dim=0) 
            batch_results[i%N][1] = torch.cat((batch_results[i%N][1], instance_bboxes), dim=0) 
            batch_results[i%N][2] = torch.cat((batch_results[i%N][2], instance_labels), dim=0) 
            batch_results[i%N][3] = torch.cat((batch_results[i%N][3], instance_scores), dim=0) 
            batch_results[i%N][4][:,y1:y2,x1:x2] += semantic_maskes.float()

            if i % N == 0:
                count_map[:,y1:y2,x1:x2] += 1

     # Finalize predictions by applying non-max suppression and other adjustments
    for i in range(N):
        batch_results[i][0], batch_results[i][1], batch_results[i][2], batch_results[i][3] = non_max_suppression_multiclass(\
            batch_results[i][0], batch_results[i][1], batch_results[i][2], batch_results[i][3], threshold=0.5)

        batch_results[i][4] = (batch_results[i][4] / count_map) > 0.5
        batch_results[i] = get_instance_segmentation_results(batch_results[i], image_bbox=batch["img_bboxes"][i], class_names=batch["class_names"][i])

    return batch_results


def evaluate(args, model, dataloader, max_num=5, save_path=None, stage="test"):
    """
    Evaluates a model on a dataset using provided dataloader. It supports saving predictions and computing evaluation metrics.

    Parameters:
    - model: The neural network model to be evaluated.
    - dataloader: A DataLoader object that provides batches of data for evaluation.
    - max_num: Maximum number of batches to process during evaluation. Defaults to 5.
    - save_path: Directory path where images with predictions will be saved. If None, images are not saved.
    - stage: Evaluation stage ("train" or "test"). Determines how predictions are made (image patch vs sliding window).
    
    Returns:
    - loss: Average loss over the processed batches.
    - bbox_mAP: Mean Average Precision for bounding boxes.
    - mask_mAP: Mean Average Precision for segmentation masks.
    - mask_mF1: Mean F1 score for all classes at all IoU thresholds.
    - mIoU: Mean Intersection over Union for semantic segmentation.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Extract categories from the first batch for ground truth initialization
    for batch in dataloader:
        categories = batch["coco_datas"][0]["categories"]
        break
    gt = {"images": [], "annotations": [], "categories": categories}
    dt = [] # Detection results
    
     # Color palette for instance and semantic segmentation visualization
    instance_palette = get_instance_palette(5000)
    semantic_palette = get_instance_palette(len(gt["categories"]) + 2)
    
    # Initialize arrays for storing true positives, false positives, etc.
    semantic_true_nums = [0 for _ in range(len(gt["categories"]))]
    semantic_pred_nums = [0 for _ in range(len(gt["categories"]))]
    semantic_correct_nums = [0 for _ in range(len(gt["categories"]))]

    labels = [i+1 for i in range(len(categories))]  # Labels start from 1
    iou_thres = np.linspace(0.5, 0.95, 10).tolist() # IoU thresholds
    print("ious:", iou_thres)
    
    all_true_nums  = [0 for _ in range(len(categories))]
    all_pred_nums = [0 for _ in range(len(categories))]
    all_tp_nums = [[0 for _ in range(len(iou_thres))] for _ in range(len(categories))]

    save_num = 50  # Number of images to save visualizations for
    img_num  = 0   # Counter for processed images
    
    # Ensure max_num does not exceed dataset size
    max_num = min(max_num, len(dataloader))
    loss = 0
    num = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()
            
            num +=1
            if num > max_num:
                break
            
            # Move tensors to GPU if available
            batch['pixel_values'] = [item.cuda() for item in batch['instance_masks']] if isinstance(batch['pixel_values'], list) else batch['pixel_values'].cuda() 
            batch['pixel_mask']   = [item.cuda() for item in batch['pixel_mask']] if isinstance(batch['pixel_mask'], list) else batch['pixel_mask'].cuda() 
            batch['instance_masks']  = [item.cuda() for item in batch['instance_masks']]
            batch['instance_bboxes'] = [item.cuda() for item in batch['instance_bboxes']]
            batch['instance_labels'] = [item.cuda() for item in batch['instance_labels']]
            batch['semantic_masks']  = [item.cuda() for item in batch['semantic_masks']]

            # Perform prediction based on the stage
            if stage == "train":
                batch_results = predict_whole(model, batch)
            else:
                batch_results = predict_slide_window(model, batch, patch_size=args.patch_size)

            loss += outputs["loss"].item() if "loss" in batch_results else 0
            
            for i in range(batch['pixel_values'].size(0)):
                print("names:", batch["names"][i])
                
                x1, y1, x2, y2 = batch["img_bboxes"][i]
                pixel_values = batch['pixel_values'][i][:,y1:y2,x1:x2]
                pixel_mask   = batch['pixel_mask'][i][:,y1:y2,x1:x2]
                instance_masks = batch['instance_masks'][i][:,y1:y2,x1:x2]
                semantic_masks = batch['semantic_masks'][i][:-1,y1:y2,x1:x2]
                batch['instance_bboxes'][i][:,[0,2]] -= x1
                batch['instance_bboxes'][i][:,[1,3]] -= y1
                
                results = batch_results[i]
                
                if img_num < save_num:
                    # Save the original image
                    image = pixel_values.permute(1,2,0).clone().detach().cpu().numpy().astype(np.uint8)[:,:,::-1]
                    image = np.ascontiguousarray(image)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + ".jpg"))
                    PILImage.fromarray(image).save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + ".jpg"))
                    
                    # Process and save the semantic ground truth
                    image_semantic_gt = copy.deepcopy(image)
                    semantic_gt = bimask_to_instance_mask(semantic_masks).cpu().numpy()
                    color_mask = id_map_to_color(semantic_gt, semantic_palette)
                    image_semantic_gt[semantic_gt>0] = image_semantic_gt[semantic_gt>0]//2 + color_mask[semantic_gt>0]//2
                    image_semantic_gt = PILImage.fromarray(image_semantic_gt)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_semantic_gt.png"))
                    image_semantic_gt.save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_semantic_gt.png"))

                    # Process and save the semantic segmentation result
                    image_semantic_dt = copy.deepcopy(image)
                    semantic_dt = results['semantic_mask_ids'].cpu().numpy()
                    color_mask = id_map_to_color(semantic_dt, semantic_palette)
                    image_semantic_dt[semantic_dt>0] = image_semantic_dt[semantic_dt>0]//2 + color_mask[semantic_dt>0]//2
                    image_semantic_dt = PILImage.fromarray(image_semantic_dt)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_semantic_dt.png"))
                    image_semantic_dt.save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_semantic_dt.png"))
                    
                    # Process and the instance ground truth
                    image_instance_gt = copy.deepcopy(image)
                    instance_gt = bimask_to_instance_mask(instance_masks).cpu().numpy()
                    color_mask = id_map_to_color(instance_gt, instance_palette)
                    image_instance_gt[instance_gt>0] = image_instance_gt[instance_gt>0]//2 + color_mask[instance_gt>0]//2
                    
                    # Process and the instance segmentation result
                    image_instance_dt = copy.deepcopy(image)
                    instance_dt = results['instance_mask_ids'].cpu().numpy()
                    color_mask = id_map_to_color(instance_dt, instance_palette)
                    image_instance_dt[instance_dt>0] = image_instance_dt[instance_dt>0]//2 + color_mask[instance_dt>0]//2

                    # Draw bounding boxes and labels on the instance ground truth image
                    for bbox, label in zip(batch['instance_bboxes'][i],  batch['instance_labels'][i]):
                        x1, y1, x2, y2 = bbox
                        bbox = [x1, y1, x2-x1, y2-y1]
                        category_id = label + 1
                        cv2.rectangle(image_instance_gt, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color=semantic_palette[category_id*3:category_id*3+3], thickness=2)
                        class_score = batch["class_names"][i][category_id-1]
                        cv2.putText(image_instance_gt, class_score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id*3:category_id*3+3], 1)
                    
                    # Save the instance segmentation ground truth
                    image_instance_gt = PILImage.fromarray(image_instance_gt)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_instance_gt.png"))
                    image_instance_gt.save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_instance_gt.png"))

                if img_num < save_num:
                    # Create a copy of the image for category detection overlay
                    image_category_dt = copy.deepcopy(image)
                    category_dt = results['category_mask_ids'].cpu().numpy()
                    color_mask = id_map_to_color(category_dt, semantic_palette)
                    image_category_dt[category_dt>0] = image_category_dt[category_dt>0]//2 + color_mask[category_dt>0]//2
                    # Save the category detection results
                    image_category_dt = PILImage.fromarray(image_category_dt)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_category_dt.png"))
                    image_category_dt.save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_category_dt.png"))
                
                # Update counts for semantic segmentation evaluation
                for c in range(semantic_masks.size(0)):
                    semantic_true_nums[c] += semantic_masks[c].sum().item()
                    semantic_pred_nums[c] += results["semantic_maskes"][c].sum().item()
                    semantic_correct_nums[c] += (semantic_masks[c] * results["semantic_maskes"][c]).sum().item()

                # Calculate true positives, false positives, etc., for instance segmentation
                true_nums, pred_nums, tp_nums = true_positive_num_multiclass(instance_masks, batch['instance_labels'][i]+1, \
                    results['instance_maskes'],  results['instance_labels'],  results['instance_scores'], labels=labels, iou_thres=iou_thres)
                all_true_nums = [item1 + item2 for item1, item2 in zip(all_true_nums, true_nums)]
                all_pred_nums = [item1 + item2 for item1, item2 in zip(all_pred_nums, pred_nums)]
                all_tp_nums = [[m + n  for m, n in zip(item1, item2)] for item1, item2 in zip(all_tp_nums, tp_nums)]
                
                # Append ground truth information
                gt["images"]      += batch["coco_datas"][i]["images"]
                gt["annotations"] += batch["coco_datas"][i]["annotations"]

                # Process and append detection results
                for j in range(results['instance_maskes'].size(0)):
                    image_id = batch["coco_datas"][i]["images"][0]["id"]
                    category_id = results['instance_labels'][j].item()
                    score = results['instance_scores'][j].item()
                    
                    # Use mask_utils.encode() to transform mask into RLE
                    binary_mask = results['instance_maskes'][j].cpu().numpy().astype(np.uint8)
                    segmentation = mask_utils.encode(np.asfortranarray(binary_mask))
                    segmentation['counts'] = segmentation['counts'].decode('utf-8')
                    # bbox = list(mask_utils.toBbox(segmentation))
                    bbox = results['instance_bboxes'][j].cpu().numpy().tolist()
                    
                    dt.append({"image_id": image_id, 
                               "category_id": category_id,
                               "bbox": bbox, 
                               "segmentation": segmentation,
                               "score": score,
                               })

                    # Draw bounding boxes and labels on the instance segmentation image
                    if img_num < save_num and score >= 0.5:
                        cv2.rectangle(image_instance_dt, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color=semantic_palette[category_id*3:category_id*3+3], thickness=2)
                        class_score = batch["class_names"][i][category_id-1] + "-" + str(round(score, 4))
                        cv2.putText(image_instance_dt, class_score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id*3:category_id*3+3], 1)
                        
                # Save instance segmentation results
                if img_num < save_num:
                    image_instance_dt = PILImage.fromarray(image_instance_dt)
                    MakePath(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_instance_dt.png"))
                    image_instance_dt.save(os.path.join(save_path, batch["datasets"][i], batch["names"][i] + "_instance_dt.png"))

                img_num += 1


    # Compute evaluation metrics
    bbox_mAP = 0
    mask_mAP = 0
    mask_mF1 = 0
    mIoU = 0

    # Save ground truth and detection results as JSON files for COCO evaluation
    MakePath("./temp/gt.json")
    with open("./temp/gt.json", "w", encoding="utf8") as f:
        json.dump(gt, f, indent=4, ensure_ascii=False)

    MakePath("./temp/dt.json")
    with open("./temp/dt.json", "w", encoding="utf8") as f:
        json.dump(dt, f, indent=4, ensure_ascii=False)
        
    # Evaluate using COCO API
    # Object detection mAP
    try:
        cocoGt = COCO("./temp/gt.json")
        cocoDt = cocoGt.loadRes("./temp/dt.json")

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.maxDets=[40,400,4000] ##
        
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        bbox_mAP = cocoEval.stats[0]
        print("bbox_mAP:", bbox_mAP)
        print(" ")
    except:
        pass

    # Instance segmentation mAP
    try:
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.maxDets=[40,400,4000] ##
        
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mask_mAP = cocoEval.stats[0]
        print("mask_mAP:", mask_mAP)
        print(" ")
    except:
        pass
    
    # Instance segmentation mF1
    try:
        class_num = 0
        for i in range(len(categories)):
            if all_true_nums[i] == 0 and all_pred_nums[i] == 0:
                continue
            P = [item / max(all_pred_nums[i], 1) for item in all_tp_nums[i]]
            R = [item / max(all_true_nums[i], 1) for item in all_tp_nums[i]]
            F = [2*item / max(all_true_nums[i] + all_pred_nums[i], 1) for item in all_tp_nums[i]]
            print(categories[i]["name"], all_true_nums[i], all_pred_nums[i], all_tp_nums[i])
            print("precision:", P)
            print("recall:", R)
            print("f1-score:", F)
            print("mean F1:", sum(F) / len(F))
            mask_mF1 += sum(F) / len(F)
            class_num += 1
        print(" ")
        mask_mF1 /= class_num
        print("Mean F1 of all class at all IoUs:", mask_mF1)
        print(" ")
    except:
        pass
    
    # Semantic segmentation mIoU
    try:
        IoUs = [c/max(a+b-c, 1) for a, b, c in zip(semantic_true_nums, semantic_pred_nums, semantic_correct_nums) if a + b > 0]
        mIoU = sum(IoUs) / len(IoUs)
        
        print("semantic_true_nums:", semantic_true_nums)
        print("semantic_pred_nums:", semantic_pred_nums)
        print("semantic_correct_nums:", semantic_correct_nums)
        print("IoUs:", IoUs)
        print("mIoU:", mIoU)
        
    except:
        pass

    
    loss = loss / max_num
    model.train()
    torch.cuda.empty_cache()
    
    return loss, bbox_mAP, mask_mAP, mask_mF1, mIoU


def inference(args, model, data_path, max_num=5, save_path=None, stage="test"):
    """
    Inference on provided images and save predictions.

    Parameters:
    - model: The neural network model to be evaluated.
    - data_path: Data path contains images for Inference.
    - max_num: Maximum number of images to process during inference. Defaults to 5.
    - save_path: Directory path where images with predictions will be saved. If None, images are not saved.
    - stage: Evaluation stage ("train" or "test"). Determines how predictions are made (image patch vs sliding window).
    """
    
    model.eval()
    
    default_class_names = ["text", "table", "list", "title", "figure", "_background_"]
    
    with open(os.path.join(data_path, "list.txt")) as fin:
        image_list = [item.strip() for item in fin.readlines()]
    
    with torch.no_grad():
        for idx, image_name in enumerate(image_list):
            print(f"\n{idx}/{len(image_list)}: {image_name}")
            torch.cuda.empty_cache()
            
            image = cv2.imread(os.path.join(data_path, "image", image_name), cv2.IMREAD_COLOR)

            low, high = (704, 896)
            short_side = (low + high) // 2
            max_long_side = short_side * 2
            
            hei, wid = image.shape[0], image.shape[1]
            scale = short_side / min(hei, wid)
            hei = min(round(hei * scale), max_long_side)
            wid = min(round(wid * scale), max_long_side)
            image = cv2.resize(image, dsize=(wid, hei), fx=0, fy=0, interpolation=cv2.INTER_AREA)

            pixel_values = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float()
            pixel_mask = torch.ones((1, 1, pixel_values.size(2), pixel_values.size(3))).float()
            
            pixel_values = pixel_values.to(next(model.parameters()).device)
            pixel_mask = pixel_mask.to(next(model.parameters()).device)

            img_bboxes = torch.tensor([[0, 0, pixel_values.size(3), pixel_values.size(2)]])

            txt_name = os.path.splitext(image_name)[0] + ".txt"
            if os.path.isfile(os.path.join(data_path, "txt", txt_name)):
                with open(os.path.join(data_path, "txt", txt_name)) as fin:
                    class_names = [item.strip() for item in fin.readlines()]
                if len(class_names) == 0:
                    class_names = default_class_names
            else:
                class_names = default_class_names
                
            batch = {}
            batch["pixel_values"] = pixel_values
            batch["pixel_mask"]   = pixel_mask
            batch["img_bboxes"]   = img_bboxes
            batch["class_names"]  = [class_names]
            
            results = predict_slide_window(model, batch, patch_size=args.patch_size)[0]
            
            instance_palette = get_instance_palette(5000)
            semantic_palette = get_instance_palette(len(class_names) + 2)
            labels = [i+1 for i in range(len(class_names))]  # Labels start from 1
            
            # Save the original image
            datasets = class_names[-1].replace(" _background_", "")
            save_name = os.path.join(datasets, os.path.splitext(image_name)[0])
            
            MakePath(os.path.join(save_path, save_name + ".jpg"))
            cv2.imwrite(os.path.join(save_path, save_name + ".jpg"), image)
        
            # Process and save the semantic segmentation result
            image_semantic_dt = copy.deepcopy(image)
            semantic_dt = results['semantic_mask_ids'].cpu().numpy()
            color_mask = id_map_to_color(semantic_dt, semantic_palette)
            image_semantic_dt[semantic_dt>0] = image_semantic_dt[semantic_dt>0]//2 + color_mask[semantic_dt>0]//2
            MakePath(os.path.join(save_path, save_name + "_semantic_dt.jpg"))
            cv2.imwrite(os.path.join(save_path, save_name + "_semantic_dt.jpg"), image_semantic_dt)

            # Create a copy of the image for category detection overlay
            image_category_dt = copy.deepcopy(image)
            category_dt = results['category_mask_ids'].cpu().numpy()
            color_mask = id_map_to_color(category_dt, semantic_palette)
            image_category_dt[category_dt>0] = image_category_dt[category_dt>0]//2 + color_mask[category_dt>0]//2
            MakePath(os.path.join(save_path, save_name + "_category_dt.jpg"))
            cv2.imwrite(os.path.join(save_path, save_name + "_category_dt.jpg"), image_category_dt)
            
            # Process and the instance segmentation result
            image_instance_dt = copy.deepcopy(image)
            instance_dt = results['instance_mask_ids'].cpu().numpy()
            color_mask = id_map_to_color(instance_dt, instance_palette)
            image_instance_dt[instance_dt>0] = image_instance_dt[instance_dt>0]//2 + color_mask[instance_dt>0]//2

            # Process and append detection results
            for j in range(results['instance_maskes'].size(0)):
                category_id = results['instance_labels'][j].item()
                score = results['instance_scores'][j].item()
                bbox = results['instance_bboxes'][j].cpu().numpy().tolist()
                
                if score >= 0.5:
                    cv2.rectangle(image_instance_dt, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color=semantic_palette[category_id*3:category_id*3+3], thickness=2)
                    class_score = class_names[category_id-1] + "-" + str(round(score, 4))
                    cv2.putText(image_instance_dt, class_score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, semantic_palette[category_id*3:category_id*3+3], 1)
                    
            MakePath(os.path.join(save_path, save_name + "_instance_dt.jpg"))
            cv2.imwrite(os.path.join(save_path, save_name + "_instance_dt.jpg"), image_instance_dt)
            
            dt = []
            # Process and append detection results
            for j in range(results['instance_maskes'].size(0)):
                image_id = idx
                category_id = results['instance_labels'][j].item()
                category_id = class_names[category_id-1]
                score = results['instance_scores'][j].item()
                
                # Use mask_utils.encode() to transform mask into RLE
                binary_mask = results['instance_maskes'][j].cpu().numpy().astype(np.uint8)
                segmentation = mask_utils.encode(np.asfortranarray(binary_mask))
                segmentation['counts'] = segmentation['counts'].decode('utf-8')
                # bbox = list(mask_utils.toBbox(segmentation))
                bbox = results['instance_bboxes'][j].cpu().numpy().tolist()
                
                dt.append({"image_id": image_id, 
                            "category_id": category_id,
                            "bbox": bbox, 
                            "segmentation": segmentation,
                            "score": score,
                            })
                
                with open(os.path.join(save_path, save_name + "_instance_dt.jsonl"), "w", encoding="utf-8") as f:
                    for item in dt:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    torch.cuda.empty_cache()
    
    return


def test(args, model, max_num=5, stage="test"):
    """
    Function to test a model on a given dataset.
    
    Parameters:
    - model: The trained model to evaluate.
    - args: Arguments containing paths and other settings for evaluation.
    - max_num: Maximum number of images to process. Default is 5.
    - stage: Evaluation stage, typically "test". This could be used to differentiate between validation and test phases.
    """

    datas = []
    losses = []
    bbox_mAPs = []
    mask_mAPs = []
    mask_mF1s = []
    mIoUs = []
    
    # Iterate over all evaluation data paths provided in args
    for index, data_path in enumerate(args.eval_path):
        print(f"Testing: {index} / {len(args.eval_path)} : {data_path}")

        # Extract data and sub-data identifiers from the path for logging purposes
        # data = data_path.split("/")[7]
        # subdata = data_path.split("/")[-3]
        
        segs = data_path.split("/")
        for s, seg in enumerate(segs):
            if seg in set(["Ancient", "Handwritten", "Layout", "SceneText", "Table"]):
                data = segs[s+1]
            if seg in set(["train", "test", "val"]):
                subdata = segs[s-1]
        
        data = data + " " + subdata if subdata != data else data
        datas.append(data)

        # Prepare the test dataset and loader
        test_set = DocSAM_GT([data_path], short_range=args.short_range, patch_size=args.patch_size, patch_num=args.patch_num, keep_size=args.keep_size, stage=stage)
        test_loader = DataLoaderX(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_set.collate_fn)
        
        # Evaluate the model on the current dataset and collect metrics
        loss, bbox_mAP, mask_mAP, mask_mF1, mIoU = evaluate(args, model, test_loader, max_num, save_path=args.save_path, stage=stage)
        losses.append(loss)
        bbox_mAPs.append(bbox_mAP)
        mask_mAPs.append(mask_mAP)
        mask_mF1s.append(mask_mF1)
        mIoUs.append(mIoU)
        print(" ")
        
        del test_set
        del test_loader
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    # Calculate mean metrics across all datasets tested
    mean_loss = sum(losses) / len(losses)
    mean_bbox_mAP = sum(bbox_mAPs) / len(bbox_mAPs)
    mean_mask_mAP = sum(mask_mAPs) / len(mask_mAPs)
    mean_mask_mF1 = sum(mask_mF1s) / len(mask_mF1s)
    mean_mIoU = sum(mIoUs) / len(mIoUs)
    
    # Print the metrics for each dataset as well as the overall means
    print("mAPs and mF1s:\n")
    for data, bbox_mAP, mask_mAP, mask_mF1, mIoU in zip(datas, bbox_mAPs, mask_mAPs, mask_mF1s, mIoUs):
        print(data + ": bbox_mAP: " + str(bbox_mAP) + ", mask_mAP: " + str(mask_mAP) + ", mask_mF1: " + str(mask_mF1) + ", mIoU: " + str(mIoU)) 
    print("mean_loss:", mean_loss, "mean_bbox_mAP:", mean_bbox_mAP, "mean_mask_mAP:", mean_mask_mAP, "mean_mask_mF1:", mean_mask_mF1, "mean_mIoU:", mean_mIoU)
    
    return mean_loss, mean_bbox_mAP, mean_mask_mAP, mean_mask_mF1, mean_mIoU

 
def count_parameters(model): 
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        float: Number of trainable parameters in millions.
    """
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    
    return params / 1e6


def load_para_weights(model, restore_from):
    """
    Loads pretrained weights into the model.

    Args:
        model (torch.nn.Module): PyTorch model.
        restore_from (str): Path to the pretrained model file.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    
    pre_dict = torch.load(restore_from, weights_only=True, map_location=torch.device('cpu'))
    cur_dict = model.state_dict()
    #cur_dict = {k: v for k, v in model.named_parameters() if "instance_bbox_predictor" not in k}
    
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
    print("Pretrained model loaded!!!", restore_from)
    
    return model



if __name__ == '__main__':
    start = timeit.default_timer()

    args = get_arguments()

    if args.gpus != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    
    # Instantiate the DocSAM model
    model = DocSAM(model_size=args.model_size)
    print("total paras:", count_parameters(model))
    
    # Load pretrained weights into the model if a valid path is provided in 'args.restore_from'
    if os.path.isfile(args.restore_from):
        model = load_para_weights(model, args.restore_from)
    model.cuda()
    
    # Evaluate the model using the test() function. It takes the model, arguments, maximum number of images to process, and the stage.
    if args.stage == "test":
        print(args.eval_path)
        mean_loss, mean_bbox_mAP, mean_mask_mAP, mean_mask_mF1, mean_mIoU = test(args, model, max_num=args.max_num, stage="test")
        print("mean_loss:", mean_loss, "mean_bbox_mAP:", mean_bbox_mAP, "mean_mask_mAP:", mean_mask_mAP, "mean_mask_mF1:", mean_mask_mF1, "mean_mIoU:", mean_mIoU)
        
    elif args.stage == "inference":
        print(args.eval_path)
        for data_path in args.eval_path:
            inference(args, model, data_path, max_num=args.max_num, save_path=args.save_path, stage="test")

    # Record the end time and calculate the total execution time.
    end = timeit.default_timer()
    print('total time:', end-start,'seconds')
