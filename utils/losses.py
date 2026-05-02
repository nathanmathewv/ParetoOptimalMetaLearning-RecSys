import torch
import torch.nn.functional as F

def get_accuracy_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

def get_user_fairness_loss(y_pred, y_true, genders):
    """
    genders: 1D tensor of same size as y_pred/y_true. e.g. 0 for Male, 1 for Female
    """
    if y_pred.numel() == 0:
        return torch.tensor(0.0).to(y_pred.device)
        
    mask_male = (genders == 0)
    mask_female = (genders == 1)
    
    mse_male = F.mse_loss(y_pred[mask_male], y_true[mask_male]) if mask_male.any() else torch.tensor(0.0).to(y_pred.device)
    mse_female = F.mse_loss(y_pred[mask_female], y_true[mask_female]) if mask_female.any() else torch.tensor(0.0).to(y_pred.device)
    
    # Absolute difference in MSE between the two groups
    if mask_male.any() and mask_female.any():
        return torch.abs(mse_male - mse_female)
    return torch.tensor(0.0).to(y_pred.device)

def get_item_fairness_loss(y_pred, y_true, is_popular):
    """
    is_popular: boolean tensor indicating if item is popular
    """
    if y_pred.numel() == 0:
        return torch.tensor(0.0).to(y_pred.device)
        
    mask_pop = is_popular
    mask_unpop = ~is_popular
    
    mse_pop = F.mse_loss(y_pred[mask_pop], y_true[mask_pop]) if mask_pop.any() else torch.tensor(0.0).to(y_pred.device)
    mse_unpop = F.mse_loss(y_pred[mask_unpop], y_true[mask_unpop]) if mask_unpop.any() else torch.tensor(0.0).to(y_pred.device)
    
    if mask_pop.any() and mask_unpop.any():
        return torch.abs(mse_pop - mse_unpop)
    return torch.tensor(0.0).to(y_pred.device)

def get_path_fairness_loss(mp_att_batch):
    """
    mp_att_batch: list or tensor of mp_att tensors across the batch.
    Shape: (batch_size, num_meta_paths)
    """
    if isinstance(mp_att_batch, list):
        mp_att_batch = torch.stack(mp_att_batch, dim=0) # (B, P)
        
    # Exposure per path is the sum over users/items
    exposure = torch.sum(mp_att_batch, dim=0) # (P,)
    
    # Variance of exposure
    return torch.var(exposure, unbiased=False)
