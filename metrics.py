
# Calculates class intersection over union for one class (modified from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py#L79)
import numpy as np

def iou2d(pred, target, cls):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).sum()  # Cast to long to prevent overflows
    union = pred_inds.sum() + target_inds.sum() - intersection
    if union == 0:  
        aux = float('nan') 
    else:
        aux = intersection / max(union, 1)
    return aux

#Calculate IoU on volume
def iou3d(pred, target, cls):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred_idx = np.zeros_like(pred)
    target_idx = np.zeros_like(target)
    intersection = np.empty((pred.shape[0],1))
    union = np.empty((pred.shape[0],1))
    for i in range(pred.shape[0]):
        pred_slice = pred[i]
        target_slice = target[i]
        pred_idx = pred_slice == cls
        target_idx = target_slice == cls   
        intersection[i] = (pred_idx[target_idx]).sum()
        union[i] = pred_idx.sum() + target_idx.sum() - intersection[i]
    inter3d = intersection.sum()
    union3d = union.sum()
    return inter3d/union3d   
        
# Calculates pixel accuracy
def pixel_acc(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total
