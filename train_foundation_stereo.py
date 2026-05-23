#!/usr/bin/env python3
import argparse
import glob
import logging
import math
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import depth_uint8_decoding, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Local training for FoundationStereo')
    parser.add_argument('--dataset_root', type=str,
                        default='./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/',
                        help='Root of the sample stereo dataset')
    parser.add_argument('--left_dir', type=str, default=None,
                        help='Optional left image directory, overrides dataset_root structure')
    parser.add_argument('--right_dir', type=str, default=None,
                        help='Optional right image directory, overrides dataset_root structure')
    parser.add_argument('--disp_dir', type=str, default=None,
                        help='Optional disparity directory, overrides dataset_root structure')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for training images (<=1.0)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop_height', type=int, default=0,
                        help='Optional crop height for training images')
    parser.add_argument('--crop_width', type=int, default=0,
                        help='Optional crop width for training images')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of update iterations in forward pass')
    parser.add_argument('--low_memory', action='store_true',
                        help='Enable low-memory mode during forward pass')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='Maximum disparity in pixels (must be divisible by 16 ideally)')
    parser.add_argument('--hidden_dims', type=str, default='96,96,96',
                        help='Comma-separated hidden dims for the model. Current local training requires constant values, e.g. 96,96,96')
    parser.add_argument('--n_downsample', type=int, default=2,
                        help='Number of downsample stages in the context encoder')
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--corr_levels', type=int, default=3)
    parser.add_argument('--vit_size', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--mixed_precision', type=int, default=1,
                        help='Enable mixed precision training')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='Validation split ratio from the dataset')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit number of samples for quick debugging')
    return parser.parse_args()


class StereoSampleDataset(Dataset):
    def __init__(self, left_paths, right_paths, disp_paths, scale=1.0, crop_size=None):
        assert len(left_paths) == len(right_paths) == len(disp_paths)
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.disp_paths = disp_paths
        self.scale = scale
        self.crop_size = crop_size

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left = imageio.imread(self.left_paths[idx])
        right = imageio.imread(self.right_paths[idx])
        disp = imageio.imread(self.disp_paths[idx])

        if left.ndim == 2:
            left = np.stack([left] * 3, axis=-1)
        if right.ndim == 2:
            right = np.stack([right] * 3, axis=-1)

        if disp.ndim == 3 and disp.shape[2] == 3:
            disp = depth_uint8_decoding(disp)
        else:
            disp = disp.astype(np.float32)

        if self.scale != 1.0:
            height, width = left.shape[:2]
            new_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
            left = cv2.resize(left, new_size, interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, new_size, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, new_size, interpolation=cv2.INTER_LINEAR) * self.scale

        if self.crop_size is not None:
            height, width = left.shape[:2]
            crop_h, crop_w = self.crop_size
            if height >= crop_h and width >= crop_w:
                top = np.random.randint(0, height - crop_h + 1)
                left = left[top:top + crop_h]
                right = right[top:top + crop_h]
                disp = disp[top:top + crop_h]
            else:
                left = cv2.resize(left, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                right = cv2.resize(right, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                disp = cv2.resize(disp, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).float()

        return {
            'left': left,
            'right': right,
            'disp': disp,
        }


def find_dataset_paths(args):
    if args.left_dir and args.right_dir and args.disp_dir:
        left_dir = args.left_dir
        right_dir = args.right_dir
        disp_dir = args.disp_dir
    else:
        root = args.dataset_root
        left_dir = os.path.join(root, 'left', 'rgb')
        right_dir = os.path.join(root, 'right', 'rgb')
        disp_dir = os.path.join(root, 'left', 'disparity')

        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir) or not os.path.isdir(disp_dir):
            raise ValueError(
                'Dataset structure not found. Expected sample layout: \n'
                '  <dataset_root>/left/rgb/*.jpg\n'
                '  <dataset_root>/right/rgb/*.jpg\n'
                '  <dataset_root>/left/disparity/*.png\n'
                'Or provide --left_dir --right_dir --disp_dir explicitly.'
            )

    left_paths = sorted(glob.glob(os.path.join(left_dir, '*')))
    right_paths = sorted(glob.glob(os.path.join(right_dir, '*')))
    disp_paths = sorted(glob.glob(os.path.join(disp_dir, '*')))

    if len(left_paths) == 0 or len(right_paths) == 0 or len(disp_paths) == 0:
        raise ValueError('No files found in dataset directories.')

    if len(left_paths) != len(right_paths) or len(left_paths) != len(disp_paths):
        raise ValueError('Left/right/disparity counts do not match.')

    return left_paths, right_paths, disp_paths


def build_model(args):
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    if len(hidden_dims) != args.n_gru_layers:
        raise ValueError(f'hidden_dims length ({len(hidden_dims)}) must equal n_gru_layers ({args.n_gru_layers})')
    if len(set(hidden_dims)) != 1:
        raise ValueError('Current local training requires hidden_dims to be constant across scales, e.g. --hidden_dims 96,96,96')

    class ArgsWrapper:
        def __init__(self, d):
            self.__dict__.update(d)
        def __getitem__(self, key):
            return self.__dict__[key]
        def get(self, key, default=None):
            return self.__dict__.get(key, default)

    model_args = ArgsWrapper({
        'hidden_dims': hidden_dims,
        'n_downsample': args.n_downsample,
        'n_gru_layers': args.n_gru_layers,
        'max_disp': args.max_disp,
        'corr_radius': args.corr_radius,
        'corr_levels': args.corr_levels,
        'mixed_precision': bool(args.mixed_precision),
        'vit_size': args.vit_size,
    })
    return FoundationStereo(model_args)


def compute_loss(pred_disp, gt_disp, valid_mask=None):
    if valid_mask is None:
        valid_mask = gt_disp > 0
    if valid_mask.sum() == 0:
        return None
    loss = torch.abs(pred_disp[valid_mask] - gt_disp[valid_mask]).mean()
    return loss


def collate_fn(batch):
    left = torch.stack([item['left'] for item in batch], dim=0)
    right = torch.stack([item['right'] for item in batch], dim=0)
    disp = torch.stack([item['disp'] for item in batch], dim=0)
    return {'left': left, 'right': right, 'disp': disp}


def save_checkpoint(state, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, path)
    logging.info(f'Saved checkpoint: {path}')


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d|%H:%M:%S')

    logging.info('Building dataset paths...')
    left_paths, right_paths, disp_paths = find_dataset_paths(args)
    if args.max_samples > 0:
        left_paths = left_paths[: args.max_samples]
        right_paths = right_paths[: args.max_samples]
        disp_paths = disp_paths[: args.max_samples]
    dataset = StereoSampleDataset(
        left_paths,
        right_paths,
        disp_paths,
        scale=args.scale,
        crop_size=(args.crop_height, args.crop_width) if args.crop_height and args.crop_width else None,
    )

    val_loader = None
    if args.val_ratio > 0.0:
        split = int(len(dataset) * (1.0 - args.val_ratio))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        logging.info(f'Dataset split: {len(train_dataset)} train / {len(val_dataset)} val')
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        logging.info(f'Dataset size: {len(dataset)} samples')

    logging.info('Building model...')
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.mixed_precision))

    start_epoch = 0
    if args.resume_ckpt is not None:
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
        start_epoch = checkpoint.get('epoch', 0) + 1
        logging.info(f'Resuming from checkpoint {args.resume_ckpt} at epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        model.train()
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            gt_disp = batch['disp'].to(device)
            padder = InputPadder(left.shape, divis_by=32, force_square=False)
            left, right = padder.pad(left, right)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=bool(args.mixed_precision)):
                init_disp, disp_preds = model(left, right, iters=args.iters, low_memory=args.low_memory)
                if not isinstance(disp_preds, (list, tuple)):
                    pred_disp = disp_preds
                else:
                    pred_disp = disp_preds[-1]
                pred_disp = padder.unpad(pred_disp).squeeze(1)
                loss = compute_loss(pred_disp, gt_disp)

            if loss is None:
                logging.warning(f'Empty valid disparity mask for batch {batch_idx}, skipping')
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % 5 == 0:
                logging.info(f'Epoch {epoch + 1}/{args.epochs} | step {batch_idx + 1}/{len(train_loader)} | loss {loss.item():.6f}')

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
        else:
            avg_loss = float('nan')

        elapsed = time.time() - start_time
        logging.info(f'Epoch {epoch + 1} completed | avg loss {avg_loss:.6f} | time {elapsed:.1f}s')

        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }, args.output_dir, epoch + 1)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    left = val_batch['left'].to(device)
                    right = val_batch['right'].to(device)
                    gt_disp = val_batch['disp'].to(device)
                    padder = InputPadder(left.shape, divis_by=32, force_square=False)
                    left, right = padder.pad(left, right)
                    with torch.amp.autocast('cuda', enabled=bool(args.mixed_precision)):
                        _, disp_preds = model(left, right, iters=args.iters, low_memory=args.low_memory)
                        pred_disp = disp_preds[-1]
                        pred_disp = padder.unpad(pred_disp).squeeze(1)
                        loss = compute_loss(pred_disp, gt_disp)
                    if loss is None:
                        continue
                    val_loss += loss.item()
                    val_steps += 1
            if val_steps > 0:
                logging.info(f'Validation loss: {val_loss / val_steps:.6f}')
            model.train()

    logging.info('Training finished.')


if __name__ == '__main__':
    main()
