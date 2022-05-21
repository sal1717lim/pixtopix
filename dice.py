import torch

class diceloss(torch.nn.Module):
    def __init__(self):
        super(diceloss,self).__init__()
    def forward(self,pred,target):
        pred=(pred>0.3).float()
        target = (target > 0.3).float()
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1).float()  # Flatten
        m2 = target.view(num, -1).float()  # Flatten
        intersection = (m1 * m2).sum().float()
        d=(1-(2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
        print(d)
        return d
