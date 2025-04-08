# -*- coding: UTF-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.dont_write_bytecode = True

import warnings
warnings.filterwarnings("ignore")

import time
import timeit
import datetime
import random

import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch.multiprocessing as mp

from datasets.dataset import DocSAM_GT
from models.DocSAM import DocSAM 
from test import evaluate_all_datasets


MODEL_SIZE = "base"
SAVE_PATH  = './outputs/outputs_train/'

SHORT_RANGE = (704, 896)
PATCH_SIZE = (640, 640)
PATCH_NUM = 1
KEEP_SIZE = False
MAX_NUM = 10

BATCH_SIZE = 8
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULER = "cosine"

FINE_TUNE = 'false'
RESTORE_FROM = './snapshots/last_model.pth'
SNAPSHOT_DIR = './snapshots/'
START_ITER = 0
TOTAL_ITER = 1000
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
    Parses command-line arguments for configuring the DocSAM Network.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description="DocSAM Network")
    
    parser.add_argument('--model-size', type=str, default=MODEL_SIZE, help='Model size: tiny, small, base, large.') 
    
    parser.add_argument('--train-path', nargs='+', type=str, help='A list of paths to training datasets') 
    parser.add_argument("--eval-path", nargs='+', type=str, help='A list of paths to evaluation datasets') 
    parser.add_argument("--save-path", type=str, default=SAVE_PATH, help='Path to save the model predicts')
    
    parser.add_argument("--short-range", type=parse_tuple, default=SHORT_RANGE, help='Short side range')
    parser.add_argument("--patch-size", type=parse_tuple, default=PATCH_SIZE, help='Patch size sampled from each image during training')
    parser.add_argument("--patch-num", type=int, default=PATCH_NUM, help='Patch number')
    parser.add_argument("--keep-size", type=str2bool, default=KEEP_SIZE, help='Whether to keep original image size')
    parser.add_argument('--max-num', type=int, default=MAX_NUM, help='Max image num for evaluation.')  
    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help='Number of samples per batch')
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help='Initial learning rate for the optimizer')
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help='Momentum factor for the optimizer')
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help='Weight decay (L2 penalty)')
    parser.add_argument("--lr-scheduler", type=str, default=LR_SCHEDULER, help='Type of learning rate scheduler')

    parser.add_argument("--fine-tune", type=str2bool, default=FINE_TUNE, help='Whether to fine-tune the model')
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help='Path to a pre-trained model to restore from')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help='Directory to save snapshots')
    parser.add_argument("--start-iter", type=int, default=START_ITER, help='Starting iteration number')
    parser.add_argument("--total-iter", type=int, default=TOTAL_ITER, help='Total number of iterations')

    parser.add_argument("--gpus", type=str, default=GPU_IDS, help='Comma-separated list of GPU IDs to use')

    return parser.parse_args()


def seed_torch(seed=1029):
    """
    Sets seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value to set. Default is 1029.
    """
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MakePath(path):
    """
    Creates directories if they do not exist.

    Args:
        path (str): Path to create.
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


def adjust_learning_rate_cos(optimizer, cur_iter=0, max_iter=100, warm_up=10, base_lr=1e-2, power=1.0, T_max=100):
    """
    Adjusts the learning rate using a cosine annealing schedule with warm-up.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate needs adjustment.
        cur_iter (int): Current iteration number.
        max_iter (int): Maximum number of iterations.
        warm_up (int): Number of warm-up iterations.
        base_lr (float): Base learning rate.
        power (float): Power factor for the learning rate adjustment.
        T_max (int): Maximum number of iterations for one cycle.

    Returns:
        float: The new learning rate.
    """
    
    if cur_iter < warm_up:
        lr = cur_iter / warm_up * base_lr
    else: 
        cur_iter -= warm_up
        #lr = base_lr * (1 - (cur_iter // T_max * T_max) / max_iter) ** power
        lr = base_lr * (pow(0.2, cur_iter // T_max))
        lr *= 0.5 + 0.5 * math.cos(cur_iter % T_max / T_max * math.pi)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_step(optimizer, cur_iter=0, max_iter=100, warm_up=10, base_lr=1e-2, steps=[0, 10, 20, 30]):
    """
    Adjusts the learning rate using a step decay schedule with warm-up.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate needs adjustment.
        cur_iter (int): Current iteration number.
        max_iter (int): Maximum number of iterations.
        warm_up (int): Number of warm-up iterations.
        base_lr (float): Base learning rate.
        steps (list): List of iteration numbers where the learning rate should drop.

    Returns:
        float: The new learning rate.
    """
    
    if steps is None:
        steps = [0, max_iter//2, 3*max_iter//4, max_iter]
    if cur_iter < warm_up:
        lr = cur_iter / warm_up * base_lr
    else: 
        cur_iter -= warm_up
        for i, step in enumerate(steps):
            if cur_iter >= step:
                lr = base_lr * pow(0.1, i)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr


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

    # Parse command-line arguments
    args = get_arguments()

    # Initialize the DocSAM model
    model = DocSAM(model_size=args.model_size)
    
    # # Optionally freeze certain parameters
    # for name, param in model.named_parameters():
    #    if "textual_encoder" in name:
    #        param.requires_grad = False
           
    print("total paras:", count_parameters(model))

    # Load pretrained weights if fine-tuning is enabled
    if args.fine_tune == True:
        if os.path.isfile(args.restore_from):
            model = load_para_weights(model, args.restore_from)
    else:
        args.start_iter = 0
    
    # Define optimizer
    params = model.parameters()

    #optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize distributed training environment
    #os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model.train()
    
    # Create training dataset and loader
    train_set = DocSAM_GT(args.train_path, short_range=args.short_range, patch_size=args.patch_size, patch_num=args.patch_num, keep_size=args.keep_size, stage="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(train_set, batch_size=int(args.batch_size/len(args.gpus.split(','))), num_workers=4, pin_memory=True, sampler=train_sampler, collate_fn=train_set.collate_fn)
    
    # Create snapshot directory if it does not exist
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        
    # Calculate number of iterations per epoch and total epochs
    num_iters = len(train_loader)
    num_epoch = math.ceil(args.total_iter / num_iters)
    if local_rank == 0:
        print(datetime.datetime.now(), f'INFO: MAE: Start training, Total images: {len(train_loader)*args.batch_size}, Total iters: {args.total_iter}, Total epochs: {num_epoch}, Iters of each epoch: {num_iters}') 
    start_time = time.time()  
    
    if local_rank == 0:
        # Evaluate initial performance metrics
        max_bbox_mAP, max_mask_mAP, max_mask_mF1, max_mIoU = evaluate_all_datasets(args, model, stage="train") 
        
    flag = True
    for i_epoch in range(num_epoch):
        if flag==False:
            break
        train_sampler.set_epoch(i_epoch)
        for i_iter, batch in enumerate(train_loader, start=1):
            current_iter = args.start_iter + i_epoch*num_iters + i_iter
            
            # Zero gradients, forward pass, backward pass, and optimization step
            optimizer.zero_grad()
            preds = model(batch)
            loss = preds["loss"]
            loss.backward()
            optimizer.step()

            # Adjust learning rate based on the scheduler type
            if args.lr_scheduler == "cosine":
                lr = adjust_learning_rate_cos(optimizer, current_iter, args.total_iter, 0, args.learning_rate, power=1.0, T_max=args.total_iter)
            elif args.lr_scheduler == "step":
                steps = [0, args.total_iter//2, 3*args.total_iter//4, args.total_iter]
                lr = adjust_learning_rate_step(optimizer, current_iter, args.total_iter, 0, args.learning_rate, steps=steps)
            else:
                steps = [0, args.total_iter]
                lr = adjust_learning_rate_step(optimizer, current_iter, args.total_iter, 0, args.learning_rate, steps=steps)
            
            # Calculate elapsed time and estimated time to completion
            time_cost = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            eta_seconds = ((time.time() - start_time) / (current_iter-args.start_iter)) * (args.total_iter - current_iter)
            time_need = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            # Print progress every 10 iterations if rank is 0
            if local_rank == 0 and current_iter % 10 == 0:
                print(datetime.datetime.now(), "INFO:HisDoc2022: Iters: {:d}/{:d} || Lr: {:.8f} || Loss: {:.4f} || Time: {} <-- {}".format(current_iter, args.total_iter, lr, loss.item(), time_cost, time_need))

            # Save snapshots and update best model if necessary
            if current_iter % 200 == 0 or current_iter == args.total_iter:
                if local_rank == 0:
                    bbox_mAP, mask_mAP, mask_mF1, mIoU = evaluate_all_datasets(args, model, stage="train") 
                    if mask_mAP > max_mask_mAP:
                        print(datetime.datetime.now(), 'Best model updated, mAP: {:.4f}-->{:.4f}, taking snapshot...'.format(max_mask_mAP, mask_mAP))
                        torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, 'best_model.pth'))
                        max_mask_mAP = mask_mAP
                    torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, 'last_model.pth'))
                    print(datetime.datetime.now(), 'Best model mAP: {:.4f},'.format(max_mask_mAP))
                torch.cuda.empty_cache()
            
            # Stop training if total iterations reached
            if current_iter > args.total_iter:
                flag = False
                torch.cuda.empty_cache()
                break

    end = timeit.default_timer()
    print('total time:', end-start,'seconds')

    
