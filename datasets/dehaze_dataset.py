"""
Dataset Loader for Image Dehazing
Supports DHID, LHID, and RICE dataset formats
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class DehazeDataset(Dataset):
    """
    Image Dehazing Dataset
    
    Supports directory structures:
        - GT/ and Haze/ folders (standard format)
        - Paired images with naming conventions
    
    Supports datasets: DHID, LHID, RICE, custom datasets
    """
    
    def __init__(
        self,
        root_dir,
        image_size=256,
        split='train',
        augment=True,
    ):
        """
        Args:
            root_dir: root directory containing GT and Haze folders
            image_size: target image size for training
            split: 'train', 'val', or 'test'
            augment: whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        self.augment = augment and split == 'train'
        
        # Paths to GT and Haze images
        self.gt_dir = os.path.join(root_dir, split, 'GT')
        self.haze_dir = os.path.join(root_dir, split, 'Haze')
        
        # Alternative structure (no split subdirectory)
        if not os.path.exists(self.gt_dir):
            self.gt_dir = os.path.join(root_dir, 'GT')
            self.haze_dir = os.path.join(root_dir, 'Haze')
        
        # Check if directories exist
        if not os.path.exists(self.gt_dir) or not os.path.exists(self.haze_dir):
            raise ValueError(f"GT or Haze directory not found in {root_dir}")
        
        # Get image file lists
        self.gt_images = sorted([f for f in os.listdir(self.gt_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.haze_images = sorted([f for f in os.listdir(self.haze_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Match GT and Haze images
        self.image_pairs = self._match_pairs()
        
        print(f"Loaded {len(self.image_pairs)} image pairs from {root_dir} ({split})")
    
    def _match_pairs(self):
        """Match GT and Haze images"""
        pairs = []
        
        # Try to match by filename
        gt_dict = {os.path.splitext(f)[0]: f for f in self.gt_images}
        haze_dict = {os.path.splitext(f)[0]: f for f in self.haze_images}
        
        # Match exact names
        for gt_name in gt_dict:
            if gt_name in haze_dict:
                pairs.append((gt_dict[gt_name], haze_dict[gt_name]))
        
        # If no matches, try matching by removing common suffixes/prefixes
        if len(pairs) == 0:
            for gt_file in self.gt_images:
                # Common naming patterns: GT_xxx.png <-> Haze_xxx.png
                # or xxx_gt.png <-> xxx_haze.png
                base_name = gt_file.replace('_gt', '').replace('_GT', '').replace('GT_', '')
                base_name = os.path.splitext(base_name)[0]
                
                for haze_file in self.haze_images:
                    haze_base = haze_file.replace('_haze', '').replace('_Haze', '').replace('Haze_', '')
                    haze_base = os.path.splitext(haze_base)[0]
                    
                    if base_name == haze_base:
                        pairs.append((gt_file, haze_file))
                        break
        
        # If still no matches, assume same order
        if len(pairs) == 0:
            print("Warning: Could not match GT and Haze images by name, assuming same order")
            pairs = list(zip(self.gt_images, self.haze_images))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def _load_image(self, path):
        """Load image and convert to RGB"""
        img = Image.open(path).convert('RGB')
        return img
    
    def _transform(self, gt_img, haze_img):
        """Apply transforms and augmentations"""
        
        # Resize if needed
        if gt_img.size != (self.image_size, self.image_size):
            gt_img = TF.resize(gt_img, (self.image_size, self.image_size), 
                              interpolation=Image.BICUBIC)
            haze_img = TF.resize(haze_img, (self.image_size, self.image_size), 
                                interpolation=Image.BICUBIC)
        
        # Data augmentation for training
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                gt_img = TF.hflip(gt_img)
                haze_img = TF.hflip(haze_img)
            
            # Random vertical flip
            if random.random() > 0.5:
                gt_img = TF.vflip(gt_img)
                haze_img = TF.vflip(haze_img)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                gt_img = TF.rotate(gt_img, angle)
                haze_img = TF.rotate(haze_img, angle)
        
        # Convert to tensor and normalize to [-1, 1]
        gt_img = TF.to_tensor(gt_img)
        haze_img = TF.to_tensor(haze_img)
        
        gt_img = (gt_img - 0.5) * 2.0
        haze_img = (haze_img - 0.5) * 2.0
        
        return gt_img, haze_img
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'gt': ground truth image (C, H, W) in range [-1, 1]
                'haze': hazy image (C, H, W) in range [-1, 1]
                'gt_path': path to GT image
                'haze_path': path to Haze image
        """
        gt_filename, haze_filename = self.image_pairs[idx]
        
        gt_path = os.path.join(self.gt_dir, gt_filename)
        haze_path = os.path.join(self.haze_dir, haze_filename)
        
        # Load images
        gt_img = self._load_image(gt_path)
        haze_img = self._load_image(haze_path)
        
        # Apply transforms
        gt_img, haze_img = self._transform(gt_img, haze_img)
        
        return {
            'gt': gt_img,
            'haze': haze_img,
            'gt_path': gt_path,
            'haze_path': haze_path,
        }


class DehazeDatasetWithCrop(DehazeDataset):
    """
    Dehazing dataset with random cropping
    Useful for training on high-resolution images
    """
    
    def __init__(
        self,
        root_dir,
        image_size=256,
        crop_size=None,
        split='train',
        augment=True,
    ):
        """
        Args:
            crop_size: size to crop images to (if None, use image_size)
        """
        super().__init__(root_dir, image_size, split, augment)
        self.crop_size = crop_size if crop_size is not None else image_size
    
    def _transform(self, gt_img, haze_img):
        """Apply transforms with random cropping"""
        
        # Resize to a larger size first if needed
        if gt_img.size[0] < self.image_size or gt_img.size[1] < self.image_size:
            gt_img = TF.resize(gt_img, (self.image_size, self.image_size), 
                              interpolation=Image.BICUBIC)
            haze_img = TF.resize(haze_img, (self.image_size, self.image_size), 
                                interpolation=Image.BICUBIC)
        
        # Random crop if augmenting
        if self.augment and self.crop_size < min(gt_img.size):
            i, j, h, w = transforms.RandomCrop.get_params(
                gt_img, output_size=(self.crop_size, self.crop_size)
            )
            gt_img = TF.crop(gt_img, i, j, h, w)
            haze_img = TF.crop(haze_img, i, j, h, w)
        else:
            # Center crop for validation/test
            gt_img = TF.center_crop(gt_img, (self.crop_size, self.crop_size))
            haze_img = TF.center_crop(haze_img, (self.crop_size, self.crop_size))
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                gt_img = TF.hflip(gt_img)
                haze_img = TF.hflip(haze_img)
            
            # Random vertical flip
            if random.random() > 0.5:
                gt_img = TF.vflip(gt_img)
                haze_img = TF.vflip(haze_img)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                gt_img = TF.rotate(gt_img, angle)
                haze_img = TF.rotate(haze_img, angle)
        
        # Convert to tensor and normalize
        gt_img = TF.to_tensor(gt_img)
        haze_img = TF.to_tensor(haze_img)
        
        gt_img = (gt_img - 0.5) * 2.0
        haze_img = (haze_img - 0.5) * 2.0
        
        return gt_img, haze_img
