import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Gradient_Net(nn.Module):

  def __init__(self,batchsize):


    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).to(device)
    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).to(device)
    kernel__y=torch.zeros((batchsize,1,3,3))
    kernel__x = torch.zeros((batchsize, 1, 3, 3))
    for i in range(batchsize):
        kernel__x[i,:,:,:]=kernel_x
        kernel__y[i, :, :, :] = kernel_y
    kernel__y=kernel__y.to(device)
    kernel__x = kernel__x.to(device)
    self.weight_x = nn.Parameter(data=kernel__x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel__y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x.double().to(device), self.weight_x.double().to(device))
    grad_y = F.conv2d(x.double().to(device), self.weight_y.double().to(device))
    orientation=(torch.atan2(grad_y,grad_x)*180)/torch.pi
    return orientation
def histogramme(ytrue,ypred,bins=255,range=(0.,255.)):
    h_ytrue=torch.histogram(ytrue[:,0,:,:],bins=bins,range=range)
    h_pred = torch.histogram(ypred[:,0,:,:], bins=bins, range=range)

    mse=torch.nn.MSELoss()
    distance=torch.sqrt(mse(h_ytrue[0],h_pred[0]))
    return distance,h_ytrue[0].int(),h_pred[0].int()
def orientationhist(ytrue,ypred,bins=360,range=(0.,360.),batch=1):

    d=Gradient_Net(batch)
    x=(d(ytrue)%360).int().float()
    y=(d(ypred)%360).int().float()
    return histogramme(x.unsqueeze(0),y.unsqueeze(0),bins=bins,range=range)

