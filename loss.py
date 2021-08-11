import torch
import torch.nn as nn
import torch.nn.functional as F

l1_loss_fn=torch.nn.L1Loss(reduce=True, size_average=True)

def structure_loss(pred, mask):
    size = pred.shape[2]
    mask = nn.functional.interpolate(mask, size=(size, size))

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
  
  
def dct_loss(pred,ycbcr_image,gt):
    pred = nn.functional.sigmoid(pred)

    size = pred.shape[2]

    pred=ycbcr_image*pred
    gt=ycbcr_image*gt

    num_batchsize = ycbcr_image.shape[0]
    ycbcr_pred = pred.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
    dct_pred = DCT.dct_2d(ycbcr_pred, norm='ortho')
    dct_pred = dct_pred.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)

    ycbcr_gt = gt.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
    dct_gt = DCT.dct_2d(ycbcr_gt, norm='ortho')
    dct_gt = dct_gt.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)


    dct_loss=l1_loss_fn(dct_pred,dct_gt)
    return dct_loss
