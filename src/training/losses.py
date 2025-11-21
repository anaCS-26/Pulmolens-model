import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al.)
    
    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Uses different margins for positive and negative samples.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al.)
    
    Args:
        gamma_neg: Focusing parameter for negative samples
        gamma_pos: Focusing parameter for positive samples
        clip: Probability clipping value (prevents log(0))
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Probabilities
        probs = torch.sigmoid(inputs)
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip, max=1 - self.clip)
        
        # Calculate positive and negative loss separately
        # Positive samples
        pos_loss = targets * torch.log(probs)
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        
        # Negative samples
        neg_loss = (1 - targets) * torch.log(1 - probs)
        neg_loss = neg_loss * (probs ** self.gamma_neg)
        
        # Combine
        loss = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-like objectives.
    Optimizes for overlap between prediction and target.
    
    Useful when combined with BCE for multi-label classification.
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        # Return Dice loss
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.
    Allows weighted combination of different objectives.
    
    Args:
        losses: Dictionary of {loss_name: (loss_fn, weight)}
    """
    def __init__(self, losses_dict):
        super(CombinedLoss, self).__init__()
        self.losses_dict = losses_dict
        
    def forward(self, inputs, targets):
        total_loss = 0
        loss_breakdown = {}
        
        for loss_name, (loss_fn, weight) in self.losses_dict.items():
            loss_value = loss_fn(inputs, targets)
            total_loss += weight * loss_value
            loss_breakdown[loss_name] = loss_value.item()
        
        # Store breakdown for logging (accessible via .loss_breakdown)
        self.loss_breakdown = loss_breakdown
        
        return total_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCE with additional class weights for handling imbalance.
    Extends PyTorch's BCEWithLogitsLoss with more flexible weighting.
    """
    def __init__(self, pos_weight=None, class_weights=None, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_type='bce', pos_weights=None, **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: One of 'bce', 'focal', 'asl', 'dice', 'combined'
        pos_weights: Positive class weights for BCE
        **kwargs: Additional arguments for specific losses
    
    Returns:
        Loss function instance
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    elif loss_type == 'weighted_bce':
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weights)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'asl':
        gamma_neg = kwargs.get('asl_gamma_neg', 4)
        gamma_pos = kwargs.get('asl_gamma_pos', 1)
        clip = kwargs.get('asl_clip', 0.05)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
    
    elif loss_type == 'dice':
        smooth = kwargs.get('dice_smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    elif loss_type == 'combined':
        # Default: BCE + Focal combination
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        
        losses_dict = {
            'bce': (bce, kwargs.get('bce_weight', 0.5)),
            'focal': (focal, kwargs.get('focal_weight', 0.5))
        }
        return CombinedLoss(losses_dict)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    batch_size, num_classes = 4, 14
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test each loss
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal': FocalLoss(),
        'ASL': AsymmetricLoss(),
        'Dice': DiceLoss(),
    }
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(inputs, targets)
        print(f"{name} Loss: {loss_value.item():.4f}")
    
    # Test combined loss
    combined = get_loss_function('combined')
    combined_loss = combined(inputs, targets)
    print(f"Combined Loss: {combined_loss.item():.4f}")
    print(f"  Breakdown: {combined.loss_breakdown}")
    
    print("\nAll loss functions working correctly!")
