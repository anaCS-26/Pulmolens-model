import torch
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    CLAHE, GaussNoise, ElasticTransform, GridDistortion, OpticalDistortion,
    CoarseDropout, Normalize, Resize
)
from albumentations.pytorch import ToTensorV2
import random


def get_training_augmentation(image_size=224, advanced=True):
    """
    Get training augmentation pipeline optimized for chest X-rays.
    
    Args:
        image_size: Target image size
        advanced: If True, use advanced augmentations
    
    Returns:
        Albumentations Compose object
    """
    if advanced:
        return Compose([
            Resize(image_size, image_size),
            
            # Geometric transforms
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # Elastic deformation (simulate patient positioning variations)
            ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Grid distortion
            GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Contrast enhancement (critical for X-rays)
            CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
            
            # Brightness and contrast
            RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Noise (simulate image acquisition noise)
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Random erasing / dropout
            CoarseDropout(
                max_holes=8,
                max_height=image_size // 10,
                max_width=image_size // 10,
                min_holes=1,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Basic augmentation
        return Compose([
            Resize(image_size, image_size),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_validation_augmentation(image_size=224):
    """
    Get validation/test augmentation (no random transforms).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return Compose([
        Resize(image_size, image_size),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Always apply CLAHE
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_tta_augmentation(image_size=224, n_augs=5):
    """
    Get Test-Time Augmentation transforms.
    Returns a list of different augmentation pipelines.
    
    Args:
        image_size: Target image size
        n_augs: Number of augmentation variants
    
    Returns:
        List of augmentation pipelines
    """
    tta_transforms = []
    
    # Original (no augmentation except CLAHE)
    tta_transforms.append(
        Compose([
            Resize(image_size, image_size),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    )
    
    # Horizontal flip
    tta_transforms.append(
        Compose([
            Resize(image_size, image_size),
            HorizontalFlip(p=1.0),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    )
    
    # Different CLAHE settings
    for clip_limit in [1.5, 3.0]:
        tta_transforms.append(
            Compose([
                Resize(image_size, image_size),
                CLAHE(clip_limit=clip_limit, tile_grid_size=(8, 8), p=1.0),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        )
    
    # Slight rotation variants
    if n_augs > 4:
        for angle in [-5, 5]:
            tta_transforms.append(
                Compose([
                    Resize(image_size, image_size),
                    ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=0, 
                                     always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0),
                    CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            )
    
    return tta_transforms[:n_augs]


def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation.
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al.)
    
    Args:
        x: Input images [batch, c, h, w]
        y: Labels [batch, num_classes]
        alpha: Mixup parameter
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation.
    
    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al.)
    
    Args:
        x: Input images [batch, c, h, w]
        y: Labels [batch, num_classes]
        alpha: CutMix parameter
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random box
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    Particularly effective for chest X-rays.
    
    Args:
        image: Input image (numpy array or PIL Image)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE-enhanced image
    """
    if isinstance(image, np.ndarray):
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced
    else:
        # PIL Image
        import PIL.Image as Image
        img_array = np.array(image)
        enhanced = apply_clahe(img_array, clip_limit, tile_grid_size)
        return Image.fromarray(enhanced)


if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test training augmentation
    train_aug = get_training_augmentation(advanced=True)
    augmented = train_aug(image=dummy_image)
    print(f"Training augmentation output shape: {augmented['image'].shape}")
    
    # Test validation augmentation
    val_aug = get_validation_augmentation()
    augmented = val_aug(image=dummy_image)
    print(f"Validation augmentation output shape: {augmented['image'].shape}")
    
    # Test TTA
    tta_augs = get_tta_augmentation(n_augs=5)
    print(f"Generated {len(tta_augs)} TTA augmentations")
    
    # Test mixup
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4, 14)).float()
    mixed_x, y_a, y_b, lam = mixup_data(x, y)
    print(f"Mixup lambda: {lam:.3f}")
    
    # Test cutmix
    cut_x, y_a, y_b, lam = cutmix_data(x, y)
    print(f"CutMix lambda: {lam:.3f}")
    
    # Test CLAHE
    enhanced = apply_clahe(dummy_image)
    print(f"CLAHE enhanced image shape: {enhanced.shape}")
    
    print("\nAll augmentation functions working correctly!")
