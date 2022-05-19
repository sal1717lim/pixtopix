import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from PIL import Image
from  pytorch_msssim import MS_SSIM
from costumDataset import Kaiset,depthset,RGBD
import sys
#chooses what model to train
if config.MODEL == "ResUnet":
    from resUnet import Generator
else:
    from generator_model import Generator
from matplotlib import pyplot as plt

from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
import os
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
        kernel__y[i,:,:,:]=kernel_y
        kernel__x[i,:,:,:]=kernel_x
    self.weight_x = nn.Parameter(data=kernel__x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel__y, requires_grad=False)

  def forward(self, x):
    x=x.unsqueeze(1)
    grad_x = F.conv2d(x.double(), self.weight_x.double().cuda())
    grad_y = F.conv2d(x.double(), self.weight_y.double().cuda())
    gradient =torch.sqrt( torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
    return gradient
if not os.path.exists("evaluation"):
    os.mkdir("evaluation")
g=Gradient_Net(int(sys.argv[4]))
writer=SummaryWriter("train{}-{}".format(localtime().tm_mon,localtime().tm_mday))
torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,epoch=0
):
    loop = tqdm(loader, leave=True)

    for idx, (x,x2, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            gradienttir=g(y_fake[:,0,:,:])
            gradientdepth=g(x2[:,0,:,:])


            with torch.no_grad():
                plusloss=l1_loss(gradientdepth,gradienttir)
            if sys.argv[2]=="L1":
                L1 = l1_loss(y_fake, y) * int(sys.argv[3])
            else:
                L1 = (1 - l1_loss((y_fake.type(torch.DoubleTensor) + 1) / 2, (y.type(torch.DoubleTensor) + 1) / 2)) * int(sys.argv[3])
            G_loss = G_fake_loss + L1 + 3*plusloss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            writer.add_scalar("L1 train loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real train loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake train loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )
def test_fn(
    disc, gen, loader, l1_loss, bce, epoch=0
):
    loop = tqdm(loader, leave=True)
    disc.eval()
    gen.eval()
    with torch.no_grad():
     resultat=[]
     for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2



        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            if sys.argv[2] == "L1":
                L1 = l1_loss(y_fake, y) * int(sys.argv[3])
            else:
                L1 = (1 - l1_loss((y_fake.type(torch.DoubleTensor) + 1) / 2, (y.type(torch.DoubleTensor) + 1) / 2)) * int(sys.argv[3])
            G_loss = G_fake_loss + L1
            resultat.append(L1.item())



        if idx % 10 == 0:
            writer.add_scalar("L1 test loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            writer.add_scalar("D_real test loss", torch.sigmoid(D_real).mean().item(), epoch * (len(loop)) + idx)
            writer.add_scalar("D_fake test loss", torch.sigmoid(D_fake).mean().item(), epoch * (len(loop)) + idx)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                L1    =L1.item()
            )
    disc.train()
    gen.train()
    return torch.tensor(resultat).mean()
def main():
    #instancing the models
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    #print(disc)
    gen = Generator(init_weight=config.INIT_WEIGHTS).to(config.DEVICE)
    #print(gen)
    #instancing the optims
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE*float(sys.argv[8]), betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE*float(sys.argv[8]), betas=(0.5, 0.999))
    schedulergen = torch.optim.lr_scheduler.ExponentialLR(opt_gen , gamma=0.1)
    schedulerdisc = torch.optim.lr_scheduler.ExponentialLR(opt_disc, gamma=0.1)
    #instancing the Loss-functions
    BCE = nn.BCEWithLogitsLoss()
    if sys.argv[2]=="L1":
        L1_LOSS = nn.L1Loss()
    else:
        L1_LOSS = MS_SSIM(data_range=1, size_average=True, channel=3, win_size=11)

    #if true loads the checkpoit in the ./
    if sys.argv[6]!="none":
        load_checkpoint(
            sys.argv[6], gen, opt_gen, config.LEARNING_RATE,
        )
    if sys.argv[7]!="none":
        load_checkpoint(
            sys.argv[7], disc, opt_disc, config.LEARNING_RATE,
        )

    #training data loading

    train_dataset = RGBD(path=sys.argv[1],depthpath=sys.argv[9], Listset=config.DTRAIN_LIST if sys.argv[5]=="0"else config.NTRAIN_LIST)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(sys.argv[4]),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_dataset = RGBD(path=sys.argv[1],depthpath=sys.argv[9],train=False, Listset=config.DTRAIN_LIST if sys.argv[5]=="0"else config.NTRAIN_LIST)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(sys.argv[4]),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #evauation data loading
    best=10000000
    resultat=1
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
           disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,epoch=epoch
        )
        resultat=test_fn(disc, gen, test_loader,  L1_LOSS, BCE, epoch=epoch)
        if best>resultat:
            print("improvement of the loss from {} to {}\n\n\n".format(best,resultat))
            best = resultat
        save_checkpoint(gen, opt_gen, epoch, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, epoch, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, test_loader, epoch, folder="evaluation")
        schedulergen.step()
        schedulerdisc.step()
        print("lr generateur",opt_gen.param_groups[0]["lr"])
        print("lr discriminateur", opt_gen.param_groups[0]["lr"])


if __name__ == "__main__":
    main()


