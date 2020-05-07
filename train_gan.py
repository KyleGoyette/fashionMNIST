from __future__ import print_function
#%matplotlib inline
import argparse

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from IPython.display import HTML
from PIL import Image
import wandb
import os

from models import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return vectors


def run():
    parser = argparse.ArgumentParser(description='fashion GAN')
    parser.add_argument('--log-freq', type=int, default=500)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--wd', type=float, default=1e-7)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    hyper_parameter_defaults = dict(
        nz=100,
        batch_size=8,
        learning_rate=0.0002,
        n_epochs=5,
        weight_decay=1e-6,
        beta1=0.5,
        beta2=0.999
    )



    wandb.init(project="fashion-mnist-gan", config=hyper_parameter_defaults)
    config = wandb.config

    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
    ])
    train_set = dset.FashionMNIST('./data',
                                  train=True,
                                  transform=transform,
                                  target_transform=None,
                                  download=True)

    train_loader = torch_data.DataLoader(train_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=2)
    device = torch.device('cuda:1')# if torch.cuda.is_available() else torch.device('cpu')
    netG = Generator(device, nc=1, nz=config.nz, ngf=64)
    netD = Discriminator(device, nc=1, ndf=64)
    netG.to(device)
    netD.to(device)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, config.nz, 1, 1)
    real_label = 1
    fake_label = 0
    iters = 0
    optimizerD = optim.Adam(netD.parameters(),
                            lr=config.learning_rate,
                            betas=(config.beta1, config.beta2),
                            weight_decay=config.weight_decay)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=config.learning_rate,
                            betas=(config.beta1, config.beta2),
                            weight_decay=config.weight_decay)
    img_list = []
    G_losses = []
    D_losses = []

    for epoch in range(config.n_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config.n_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                wandb.log({'Loss_D': errD.item(),
                           'Loss_G': errG.item(),
                           'D(x)': D_x,
                           'pre update D(G(z))': D_G_z1,
                           'post update D(G(z))': D_G_z2,
                           'iter': iters})

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or \
                    ((epoch == config.n_epochs - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,
                                                 padding=2,
                                                 normalize=True))
                wandb.log({"Generator Output": [wandb.Image(img_list[-1])]})
                max_pts = min(fixed_noise.shape[0]/2, 1)
                p1s = fixed_noise[:max_pts, :, :, :]
                p2s = fixed_noise[max_pts:2*max_pts, :, :, :]

                points_list = interpolate_points(p1s, p2s, 10)
                ims = []
                for point in points_list:
                    im = netG(point).detach().cpu()
                    im =transforms.ToPILImage()(im)
                    ims.append(im)
                wandb.log({"Latent Space Traversal": [wandb.Image(im) for im in ims],
                           'iter': iters})

            iters += 1


if __name__ == '__main__':
    run()
