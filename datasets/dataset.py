import os
import random
import json
import copy
import math

import cv2
import numpy as np
import jpeg4py as jpeg
from PIL import Image
import torch
from torch.utils import data
from torch.nn import functional as F
import pycocotools.mask as mask_utils
import timeit


def BboxToPolygon(bbox):
    l, r, t, b = bbox

    poly = np.zeros((4,2), np.int32)
    poly[0, :] = [l, t]
    poly[1, :] = [r, t]
    poly[2, :] = [r, b]
    poly[3, :] = [l, b]
    
    return poly


def PolygonToBBox(polygons):
    l, r, t, b = 1e10, -1, 1e10, -1
    for polygon in polygons:
        polygon = np.array(polygon).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(polygon)
        l = min(l, x)
        r = max(r, x+w)
        t = min(t, y)
        b = max(b, y+h)
    x, y, w, h = l, t, r-l, b-t

    return [x, y, w, h]


class DocSAM_GT(data.Dataset):
    def __init__(self, data_paths, short_range=(704, 896), patch_size=(640, 640), patch_num=1, keep_size=False, stage="train"):
        """
        Initializes the dataset.
        
        Parameters:
        - data_paths: list of str, paths to the datasets.
        - short_range: tuple, range for resizing the shorter side of images.
        - patch_size: tuple, size of patches to extract (height, width).
        - patch_num: int, number of patches to sample per image.
        - stage: str, specifies whether it's 'train' or 'val' phase.
        """
        
        self.data_paths = data_paths
        self.short_range = short_range
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.keep_size = keep_size
        self.stage = stage

        self.image_names = []
        for data_path in self.data_paths:
            data_list = os.path.join(data_path, "list.txt")
            if stage == "train":
                if os.path.exists(os.path.join(data_path, "list_train.txt")):
                    data_list = os.path.join(data_path, "list_train.txt")
            else:
                if os.path.exists(os.path.join(data_path, "list_val.txt")):
                    data_list = os.path.join(data_path, "list_val.txt")
            self.image_names.append([item.strip() for item in open(data_list, encoding='utf-8')])
                
        # Change according to your data path.
        self.datasets = ["" for item in self.data_paths]
        self.subdatas = ["" for item in self.data_paths]
        
        for idx, data_path in enumerate(self.data_paths):
            segs = data_path.split("/")
            for s, seg in enumerate(segs):
                if seg in set(["Ancient", "Handwritten", "Layout", "SceneText", "Table"]):
                    self.datasets[idx] = segs[s+1]
                if seg in set(["train", "test", "val"]):
                    self.subdatas[idx] = segs[s-1]

        self.dataset_sampling_probabilities = self._class_num_of_each_dataset(self.data_paths, self.image_names)

        print("Image number of each dataset:", [len(item) for item in self.image_names])
        

        
    def __len__(self):
        """Returns the max number of images available."""
        return max([len(item) for item in self.image_names]) * len(self.image_names)
        

    def _class_num_of_each_dataset(self, data_paths, image_names):
        """
        Calculates the sampling probabilities based on the number of classes in each dataset.
        
        Parameters:
        - data_paths: list of str, paths to the datasets.
        - image_names: list of list of str, names of images in each dataset.
        
        Returns:
        - list of float, sampling probabilities for each dataset.
        """

        class_nums = [0 for _ in data_paths]
        for i, (data_path, image_name) in enumerate(zip(data_paths, image_names)):
            coco_file = os.path.join(data_path, "coco", os.path.splitext(image_name[0])[0] + ".json")
            coco_data = json.load(open(coco_file))
            categories = [item for item in coco_data["categories"] if item["name"] != "_background_"]
            class_nums[i] = float(len(categories))
        gmean = pow(np.prod(class_nums), 1/len(class_nums))
        sampling_probabilities = [pow(item/gmean, 0.5) for item in class_nums]
        sampling_probabilities = [item / sum(sampling_probabilities) for item in sampling_probabilities]

        return sampling_probabilities


    def _load_image(self, data_idx, index):
        """
        Loads an image given its index within a specific dataset.
        
        Parameters:
        - data_idx: int, index of the dataset.
        - index: int, index of the image within the dataset.
        
        Returns:
        - image: torch.Tensor, loaded image tensor.
        - mask: torch.Tensor, mask tensor.
        """

        img_path = os.path.join(self.data_paths[data_idx], "image", self.image_names[data_idx][index])
        img_path_resize = os.path.join(self.data_paths[data_idx], "image_resize", self.image_names[data_idx][index])
        if os.path.exists(img_path_resize):
            img_path = img_path_resize

        if img_path.endswith(".jpg"):
            try:
                image = jpeg.JPEG(img_path).decode()[:,:,::-1].copy()
            except:
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif img_path.endswith(".gif") or img_path.endswith(".tif"):
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)[:,:,::-1].copy()
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        mask  = torch.ones((1, image.size(1),  image.size(2))).float()

        return image, mask


    def _coco_data_rectify(self, coco_data):
        """
        Rectifies COCO annotations by cleaning up invalid entries, converting IDs, processing categories,
        handling coordinates, and ensuring valid bbox and segmentation information.
        
        Parameters:
        - coco_data: dict, containing COCO format annotations.
        
        Returns:
        - coco_data: dict, rectified COCO format annotations.
        """

        # Remove annotations without valid bbox or segmentation
        annotations = []
        for ann in coco_data["annotations"]:
            if (("bbox" not in ann) or len(ann["bbox"]) != 4) and (("segmentation" not in ann) or len(ann["segmentation"]) == 0):
                continue
            annotations.append(ann)
        coco_data["annotations"] = annotations
        
        # Convert category names to IDs
        label2id = {}
        for category in coco_data["categories"]:
            label2id[category["name"]] = category["id"]

        for j in range(len(coco_data["annotations"])):
            # Ensure segmentation is in correct format
            if "poly" in coco_data["annotations"][j]:
                coco_data["annotations"][j]["segmentation"] = [coco_data["annotations"][j]["poly"]]
            # Convert string category IDs to integer IDs
            if isinstance(coco_data["annotations"][j]["category_id"], str):
                coco_data["annotations"][j]["category_id"] = label2id[coco_data["annotations"][j]["category_id"]]
                

        # Process categories: remove specific categories and reassign IDs
        category_names = [item["name"] for item in coco_data["categories"] if item["name"] != "_background_"]
        if "table" in category_names and "text word" in category_names:
            category_names = [item for item in category_names if item != "text word"]
        categories = [{'supercategory': '', 'id': id+1, 'name': name} for id, name in enumerate(sorted(category_names))]
        
        label_mapping = {}
        for m in range(len(coco_data["categories"])):
            for n in range(len(categories)):
                if coco_data["categories"][m]["name"] == categories[n]["name"]:
                    label_mapping[coco_data["categories"][m]["id"]] = categories[n]["id"]

        annotations = []
        for j in range(len(coco_data["annotations"])):
            if coco_data["annotations"][j]["category_id"] in label_mapping:
                coco_data["annotations"][j]["category_id"] = label_mapping[coco_data["annotations"][j]["category_id"]]
                annotations.append(coco_data["annotations"][j])
                
        coco_data["categories"] = categories
        coco_data["annotations"] = annotations
        
        # Handle coordinates: ensure bounding boxes and segmentations are within image boundaries
        hei = coco_data["images"][0]["height"]
        wid = coco_data["images"][0]["width"]
        for ann in range(len(coco_data["annotations"])):
            if "bbox" in coco_data["annotations"][ann] and len(coco_data["annotations"][ann]["bbox"]) == 4:
                x, y, w, h = coco_data["annotations"][ann]["bbox"]
                l, r, t, b = x, x+w, y, y+h
                l = min(max(l, 0), wid-1)
                r = min(max(r, l), wid-1)
                t = min(max(t, 0), hei-1)
                b = min(max(b, t), hei-1)
                coco_data["annotations"][ann]["bbox"] = [l, t, r-l, b-t]
                
            if "segmentation" in coco_data["annotations"][ann] and isinstance(coco_data["annotations"][ann]["segmentation"], list):
                for seg in range(len(coco_data["annotations"][ann]["segmentation"])):
                    for p in range(len(coco_data["annotations"][ann]["segmentation"][seg]) // 2):
                        coco_data["annotations"][ann]["segmentation"][seg][p*2+0] = max(0, coco_data["annotations"][ann]["segmentation"][seg][p*2+0])
                        coco_data["annotations"][ann]["segmentation"][seg][p*2+0] = min(wid-1, coco_data["annotations"][ann]["segmentation"][seg][p*2+0])
                        coco_data["annotations"][ann]["segmentation"][seg][p*2+1] = max(0, coco_data["annotations"][ann]["segmentation"][seg][p*2+1])
                        coco_data["annotations"][ann]["segmentation"][seg][p*2+1] = min(hei-1, coco_data["annotations"][ann]["segmentation"][seg][p*2+1])
        
        # Ensure annotations have both bbox and segmentation if one is missing
        for ann in range(len(coco_data["annotations"])):
            is_bbox = ("bbox" in coco_data["annotations"][ann]) and len(coco_data["annotations"][ann]["bbox"]) == 4
            is_seg_list = ("segmentation" in coco_data["annotations"][ann]) and isinstance(coco_data["annotations"][ann]["segmentation"], list) \
                and len(coco_data["annotations"][ann]["segmentation"]) > 0
                
            if (not is_bbox) and is_seg_list:
                coco_data["annotations"][ann]["bbox"] = PolygonToBBox(coco_data["annotations"][ann]["segmentation"])
            elif is_bbox and (not is_seg_list):
                x, y, w, h = coco_data["annotations"][ann]["bbox"]
                l, r, t, b = x, x+w, y, y+h
                poly = BboxToPolygon([l, r, t, b])
                coco_data["annotations"][ann]["segmentation"] = [poly.reshape(-1).tolist()]

        # Convert all IDs to strings
        for img in range(len(coco_data["images"])):
            coco_data["images"][img]["id"] = str(coco_data["images"][img]["id"])
        
        for ann in range(len(coco_data["annotations"])):
            coco_data["annotations"][ann]["id"] = str(coco_data["annotations"][ann]["id"])
            coco_data["annotations"][ann]["image_id"] = str(coco_data["annotations"][ann]["image_id"])

        return coco_data
    

    def _coco_data_reszie(self, coco_data, dsize):
        """
        Resizes COCO annotations according to the specified dimensions.
        
        Parameters:
        - coco_data: dict, containing COCO format annotations.
        - dsize: tuple, desired output size (height, width).
        
        Returns:
        - coco_data: dict, resized COCO format annotations.
        """

        hei, wid = dsize[0], dsize[1]
        hei_ori = max(coco_data["images"][0]["height"], 1)
        wid_ori = max(coco_data["images"][0]["width"], 1)
        coco_data["images"][0]["height"] = hei
        coco_data["images"][0]["width"]  = wid
        for ann in range(len(coco_data["annotations"])):
            bbox = coco_data["annotations"][ann]["bbox"]
            coco_data["annotations"][ann]["bbox"] = [bbox[0]*wid/wid_ori, bbox[1]*hei/hei_ori, bbox[2]*wid/wid_ori, bbox[3]*hei/hei_ori,]
            segmentation = coco_data["annotations"][ann]["segmentation"]
            for seg in range(len(segmentation)):
                for p in range(len(segmentation[seg]) // 2):
                    segmentation[seg][p*2+0] = segmentation[seg][p*2+0] * wid/wid_ori
                    segmentation[seg][p*2+1] = segmentation[seg][p*2+1] * hei/hei_ori
            coco_data["annotations"][ann]["segmentation"] = segmentation
            poly = [np.array(item, np.float32).reshape(-1, 2) for item in segmentation]
            area = sum([cv2.contourArea(item) for item in poly])
            coco_data["annotations"][ann]["area"] = area
            
        return coco_data


    def _load_label(self, data_idx, index, dsize):
        """
        Loads COCO format annotation for an image, rectifies and resizes it.
        
        Parameters:
        - data_idx: int, index of the dataset path.
        - index: int, index of the image within the dataset.
        - dsize: tuple(int, int), desired output size (height, width).
        
        Returns:
        - coco_data: dict, processed and resized COCO annotations.
        """

        hei, wid = dsize[0], dsize[1]
        coco_data = json.load(open(os.path.join(self.data_paths[data_idx], "coco", os.path.splitext(self.image_names[data_idx][index])[0] + '.json')))
        coco_data = self._coco_data_rectify(coco_data)
        coco_data = self._coco_data_reszie(coco_data, (hei, wid))
        coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda anno: anno["area"], reverse=True)

        return coco_data


    def _random_crop(self, image, mask, coco_data, dsize):
        """
        Applies random cropping to image, mask, and adjusts COCO annotations accordingly.
        
        Parameters:
        - image: Tensor, input image tensor.
        - mask: Tensor, input mask tensor.
        - coco_data: dict, COCO format annotations.
        - dsize: tuple(int, int), desired output size (height, width).
        
        Returns:
        - image: Tensor, cropped image.
        - mask: Tensor, cropped mask.
        - coco_data: dict, updated COCO annotations after cropping.
        """

        top  = random.randint(0, max(image.size(1)-dsize[0], 0))
        left = random.randint(0, max(image.size(2)-dsize[1], 0))
        bottom = min(top + dsize[0], image.size(1)) 
        right = min(left + dsize[1], image.size(2))
        image = image[..., top:bottom, left:right]
        mask  = mask[...,  top:bottom, left:right]

        coco_data["images"][0]["height"] = bottom - top
        coco_data["images"][0]["width"]  = right - left
        anns = []
        for ann in coco_data["annotations"]:
            x, y, w, h = ann["bbox"]
            l, r, t, b = x, x+w, y, y+h
            if l > right - 4 or r < left + 4 or t > bottom - 4 or b < top + 4:
                continue

            l, r, t, b = max(l, left) - left, min(r, right) - left, max(t, top) - top, min(b, bottom) - top
            x, y, w, h = l, t, r - l, b - t
            ann["bbox"] = [x, y, w, h]
            
            for seg in range(len(ann["segmentation"])):
                for p in range(len(ann["segmentation"][seg]) // 2):
                    ann["segmentation"][seg][p*2+0] = min(max(ann["segmentation"][seg][p*2+0], left), right) - left
                    ann["segmentation"][seg][p*2+1] = min(max(ann["segmentation"][seg][p*2+1], top), bottom) - top
                    
            poly = [np.array(item, np.float32).reshape(-1, 2) for item in ann["segmentation"]]
            area = sum([cv2.contourArea(item) for item in poly])
            ann["area"] = area
            anns.append(ann)

        coco_data["annotations"] = anns
        
        return image, mask, coco_data
    
    
    def _data_resize(self, image, mask, coco_data, dsize=None):
        """
        Resizes the input image, mask, and updates COCO annotations accordingly.
        
        Parameters:
        - image: Tensor, input image tensor.
        - mask: Tensor, input mask tensor.
        - coco_data: dict, COCO format annotations.
        - dsize: tuple(int, int) or None, desired output size (height, width).
        
        Returns:
        - image: Tensor, resized image.
        - mask: Tensor, resized mask.
        - coco_data: dict, updated COCO annotations after resizing.
        """

        if dsize is not None:
            hei, wid = dsize[-2], dsize[-1]
            
        elif self.keep_size:
            hei, wid = image.shape[-2], image.shape[-1]
            if self.stage == "train":
                hei = round(hei * pow(2, random.uniform(-0.2, 0.2)))
                wid = round(wid * pow(2, random.uniform(-0.2, 0.2)))

            low, high = self.short_range
            if min(hei, wid) < low:
                scale = low / min(hei, wid)
            elif min(hei, wid) > high:
                scale = high / min(hei, wid)
            else:
                scale = 1.0
                
            hei, wid = round(hei * scale), round(wid * scale)
            hei, wid = min(hei, min(hei, wid) * 2), min(wid, min(hei, wid) * 2)
            
        else:
            low, high = self.short_range
            if self.stage == "train":
                short_side = random.randint(low, high)
            else:
                short_side = (low + high) // 2
            max_long_side = short_side * 2
            
            hei, wid = image.shape[1], image.shape[2]
            scale = short_side / min(hei, wid)
            hei = min(round(hei * scale), max_long_side)
            wid = min(round(wid * scale), max_long_side)
            
        image = F.interpolate(image[None], size=(hei, wid), mode="area")[0]
        mask = F.interpolate(mask[None], size=(hei, wid), mode="nearest")[0]
        coco_data = self._coco_data_reszie(coco_data, (hei, wid))
            
        return image, mask, coco_data


    def _generate_mask(self, coco_data):
        """
        Generates masks for both semantic and instance segmentation from COCO annotations.
        
        Parameters:
        - coco_data: dict, COCO format annotations.
        
        Returns:
        - instance_masks: Tensor, generated instance masks.
        - instance_bboxes: Tensor, bounding boxes for instances.
        - instance_labels: Tensor, labels for instances.
        - semantic_masks: Tensor, generated semantic masks.
        - class_names: list[str], names of classes including background.
        - coco_data: dict, updated COCO annotations.
        """

        hei = coco_data["images"][0]["height"]
        wid = coco_data["images"][0]["width"]
        
        # Semantic segmentation masks initialization
        semantic_masks = [np.zeros([hei, wid], dtype=np.uint8) for _ in range(len(coco_data["categories"]) + 1)]
        class_names = [item["name"] for item in coco_data["categories"]] + ["_background_"]
    
        # Fill semantic masks
        for ann in range(len(coco_data["annotations"])):
            label = int(coco_data["annotations"][ann]["category_id"]) - 1
            poly = coco_data["annotations"][ann]["segmentation"]
            poly = [np.array(item, np.int32).reshape(-1, 2) for item in poly]
            for item in poly:
                if item.shape[0] == 0:
                    continue
                semantic_masks[label] = cv2.fillPoly(semantic_masks[label], [item], color=[1])
                semantic_masks[-1] = cv2.fillPoly(semantic_masks[-1], [item], color=[1])
        
        semantic_masks[-1] = 1 - semantic_masks[-1]
        semantic_masks = [torch.from_numpy(item) for item in semantic_masks]
        semantic_masks = torch.stack(semantic_masks, dim=0)
        
        # Instance segmentation masks initialization
        instance_masks, instance_bboxes, instance_labels = [], [], []

        # Fill instance masks
        for ann in range(len(coco_data["annotations"])):
            x, y, w, h = coco_data["annotations"][ann]["bbox"]
            bbox = [x, y, x+w, y+h]
            
            poly = coco_data["annotations"][ann]["segmentation"]
            poly = [np.array(item, np.int32).reshape(-1, 2) for item in poly]
            mask = np.zeros([hei, wid], dtype=np.uint8)
            for item in poly:
                if item.shape[0] == 0:
                    continue
                mask = cv2.fillPoly(mask, [item], color=[1])
            
            segmentation = mask_utils.encode(np.asfortranarray(mask))
            segmentation['counts'] = segmentation['counts'].decode('utf-8')
            coco_data["annotations"][ann]["segmentation"] = segmentation
            
            instance_masks.append(torch.from_numpy(mask))
            instance_bboxes.append(torch.tensor(bbox))
            instance_labels.append(torch.tensor([int(coco_data["annotations"][ann]["category_id"]) - 1]))
            
        if len(coco_data["annotations"]) == 0:
            instance_masks.append(torch.from_numpy(np.zeros((hei, wid), dtype=np.uint8)))
            instance_bboxes.append(torch.tensor([0, 0, 0, 0]))
            instance_labels.append(torch.tensor([0]))

        instance_masks  = torch.stack(instance_masks, dim=0)
        instance_bboxes = torch.stack(instance_bboxes, dim=0)
        instance_labels = torch.cat(instance_labels, dim=0)

        return instance_masks.float(), instance_bboxes.float(), instance_labels.long(), semantic_masks.float(), class_names, coco_data
    

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset based on the given index. Handles both training and evaluation stages.
        
        Parameters:
        - index: int, index of the item to retrieve from the dataset.
        
        Returns:
        - sample or all_samples: dict or list[dict], a dictionary containing image, mask, annotations, and other related information.
        """

        try:
            # Randomly choose a dataset path based on sampling probabilities
            data_idx = np.random.choice(len(self.data_paths), p=self.dataset_sampling_probabilities) 
            # Adjust index to fit within the range of available images in the selected dataset
            index = index % len(self.image_names[data_idx])

            # Load original image and mask
            ori_image, ori_mask = self._load_image(data_idx, index)
            # Load and process COCO annotations
            ori_coco_datas  = self._load_label(data_idx, index, (ori_image.size(1), ori_image.size(2)))
            
            if self.stage == "train":
                all_samples = []
                for p in range(self.patch_num):
                    if random.uniform(0, 1) < 0.2:
                        # Resize data to a fixed patch size
                        image, mask, coco_datas = self._data_resize(copy.deepcopy(ori_image), copy.deepcopy(ori_mask), copy.deepcopy(ori_coco_datas), dsize=self.patch_size)
                    else:
                        # Optionally resize without specifying size and apply random crop
                        image, mask, coco_datas = self._data_resize(copy.deepcopy(ori_image), copy.deepcopy(ori_mask), copy.deepcopy(ori_coco_datas), dsize=None)
                        image, mask, coco_datas = self._random_crop(image, mask, coco_datas, dsize=self.patch_size)
                    
                    # Generate masks for segmentation tasks
                    # coco_datas["annotations"] = coco_datas["annotations"][:900]
                    instance_masks, instance_bboxes, instance_labels, semantic_masks, class_names, coco_datas = self._generate_mask(coco_datas)

                    # Add dataset and subdataset names for contextual information
                    datasets = self.datasets[data_idx]
                    subdatas = self.subdatas[data_idx]
                    datasets = datasets + " " + subdatas if subdatas != datasets else datasets

                    # Update background class name with dataset information
                    # class_names = [datasets + " " + item for item in class_names]
                    class_names[-1] = datasets + " " + class_names[-1]

                    # Prepare sample dictionary with necessary information
                    sample = {}
                    sample['pixel_values'] = image
                    sample['pixel_mask']   = mask
                    sample['instance_masks']  = instance_masks
                    sample['instance_bboxes'] = instance_bboxes
                    sample['instance_labels'] = instance_labels
                    sample['semantic_masks']  = semantic_masks
                    sample['class_names'] = class_names
                    sample['coco_datas']  = coco_datas
                    sample['img_bboxes']  = [0, 0, image.size(2), image.size(1)]
                    sample['datasets']    = datasets
                    sample['names']       = os.path.splitext(self.image_names[data_idx][index])[0]
                    all_samples.append(sample)
                    
                return all_samples
                    
            else:
                # For evaluation stage, process data similarly but without patches
                image, mask, coco_datas = self._data_resize(copy.deepcopy(ori_image), copy.deepcopy(ori_mask), copy.deepcopy(ori_coco_datas), dsize=None)
                instance_masks, instance_bboxes, instance_labels, semantic_masks, class_names, coco_datas = self._generate_mask(coco_datas)

                datasets = self.datasets[data_idx]
                subdatas = self.subdatas[data_idx]
                datasets = datasets + " " + subdatas if subdatas != datasets else datasets
                # class_names = [datasets + " " + item for item in class_names]
                
                # Update background class name with dataset information
                # class_names = [datasets + " " + item for item in class_names]
                class_names[-1] = datasets + " " + class_names[-1]
                    
                sample = {}
                sample['pixel_values'] = image
                sample['pixel_mask']   = mask
                sample['instance_masks']  = instance_masks
                sample['instance_bboxes'] = instance_bboxes
                sample['instance_labels'] = instance_labels
                sample['semantic_masks']  = semantic_masks
                sample['class_names'] = class_names
                sample['coco_datas']  = coco_datas
                sample['img_bboxes']  = [0, 0, image.size(2), image.size(1)]
                sample['datasets']    = datasets
                sample['names']       = os.path.splitext(self.image_names[data_idx][index])[0]
            
                return sample
        
        except:
            # Handle exceptions by selecting another random sample
            idx = np.random.randint(0, len(self)-1)
            sample = self[idx]
            return sample


    def collate_fn(self, batch):
        """
        Custom collate function for batching samples. Ensures that all samples are padded to the same dimensions.
        
        Parameters:
        - batch: list[dict], a list of dictionaries containing samples.
        
        Returns:
        - dict, a dictionary containing stacked tensors and lists of various elements.
        """

        pixel_values, pixel_mask, instance_masks, instance_bboxes, instance_labels, semantic_masks, class_names, coco_datas, img_bboxes, datasets, names = [], [], [], [], [], [], [], [], [], [], []
        
        for sample in batch:
            if isinstance(sample, dict):
                # Append individual sample items to respective lists
                pixel_values.append(sample["pixel_values"])
                pixel_mask.append(sample["pixel_mask"])
                instance_masks.append(sample["instance_masks"])
                instance_bboxes.append(sample["instance_bboxes"])
                instance_labels.append(sample["instance_labels"])
                semantic_masks.append(sample["semantic_masks"])
                class_names.append(sample["class_names"])
                coco_datas.append(sample["coco_datas"])
                img_bboxes.append(sample["img_bboxes"])
                datasets.append(sample["datasets"])
                names.append(sample["names"])

            elif isinstance(sample, list):
                # Handle case where sample is a list of dictionaries
                for item in sample:
                    pixel_values.append(item["pixel_values"])
                    pixel_mask.append(item["pixel_mask"])
                    instance_masks.append(item["instance_masks"])
                    instance_bboxes.append(item["instance_bboxes"])
                    instance_labels.append(item["instance_labels"])
                    semantic_masks.append(item["semantic_masks"])
                    class_names.append(item["class_names"])
                    coco_datas.append(item["coco_datas"])
                    img_bboxes.append(item["img_bboxes"])
                    datasets.append(item["datasets"])
                    names.append(item["names"])
        
        
        # Padding images and masks to the largest dimensions in the batch
        max_hei = max([im.size(1) for im in pixel_values])
        max_wid = max([im.size(2) for im in pixel_values])
        for index in range(len(pixel_values)):
            row_padding = max_hei - pixel_values[index].size(1)
            col_padding = max_wid - pixel_values[index].size(2)
            p2d = (0, col_padding, 0, row_padding)

            img_bboxes[index] = [0, 0, pixel_values[index].size(2), pixel_values[index].size(1)]
            
            pixel_values[index] = torch.nn.functional.pad(pixel_values[index], p2d, "constant", 0)
            pixel_mask[index]   = torch.nn.functional.pad(pixel_mask[index], p2d, "constant", 0)
            instance_masks[index] = torch.nn.functional.pad(instance_masks[index], p2d, "constant", 0)
            semantic_masks[index] = torch.nn.functional.pad(semantic_masks[index], p2d, "constant", 0)

        pixel_values = torch.stack(pixel_values)
        pixel_mask   = torch.stack(pixel_mask)
        img_bboxes   = torch.tensor(img_bboxes)
        
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "instance_masks": instance_masks, "instance_bboxes": instance_bboxes, "instance_labels": instance_labels, \
            "semantic_masks": semantic_masks, "class_names": class_names, "coco_datas": coco_datas, "img_bboxes": img_bboxes, "datasets": datasets, "names": names}



if __name__ == '__main__':

    image_path = './datasets/coco/train/'
    dataset = DocSAM_GT([data_path], short_range=(704, 896), patch_size=(640, 640), patch_num=1, stage="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=test_set.collate_fn)
            
    for i, batch in enumerate(dataloader):
        print(i, batch["pixel_values"].size(), batch["pixel_mask"].size(), batch["instance_masks"][0].size(), batch["instance_labels"][0].size(),)