import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from models.networks import UNetEncoder, UNetDecoder, Classifier
import os
from PIL import Image
from dataset.dataset import *
from tqdm import tqdm
import random
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from models.networkDAMF import DAM_Classifier, DAMFEncoder, DAMFDecoder_l, DAMFDecoder_h, Fusion
from collections import defaultdict
import click
import cv2
def write_to_log(log_file_path, status):
	"""
		Write to the log file
	"""

	with open(log_file_path, "a") as log_file:
		log_file.write(status+'\n')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x



def test(fE, fL, fN, fH, fU, dataloader, fusion):
    ssim_scores = []
    psnr_scores = []
    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, name = data
        uw_img = Variable(uw_img)
        cl_img = Variable(cl_img, requires_grad=False)

        enc_out, enc_outs = fE(uw_img)
        I, I_c = fN(enc_out)
        fL_out, x_Hinput = fL(enc_out, enc_outs, I_c, fusion)
        fH_out = fH(x_Hinput, enc_outs)
        fU_out = fU(fL_out, fH_out)

        fU = to_img(fU_out)

        save_image(torch.stack([fU_out.squeeze().cpu().data]), 'D:/Data/4-6niqe/DAL/{}.jpg'.format(name[0]))
        save_image(fL_out.cpu().data, './results/fL_out.png')

        fU_out = (fU_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ssim = ssim_fn(fU_out, cl_img, multichannel=True)
        psnr = psnr_fn(cl_img, fU_out)

        ssim_scores.append(ssim)
        psnr_scores.append(psnr)

    return ssim_scores, psnr_scores

@click.command()
@click.argument('name', default='unet_adv')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--num_classes', default=3, help='the number of water types')
@click.option('--test_dataset', default='nyu', help='Name of the test dataset (nyu)')
@click.option('--data_path', default=None, help='Path of testing input data')
@click.option('--label_path', default=None, help='Path of testing label data')
@click.option('--test_size', default=None, help='Lambda for N loss')
@click.option('--isfusion', default=None, help='use DAM or not')
@click.option('--fe_load_path', default=None, help='Load path for pretrained fE')
@click.option('--fl_load_path', default=None, help='Load path for pretrained fL')
@click.option('--fn_load_path', default=None, help='Load path for pretrained fN')
@click.option('--fh_load_path', default=None, help='Load path for pretrained fH')
@click.option('--fu_load_path', default=None, help='Load path for pretrained fU')
def main(name, num_channels, num_classes, test_dataset, data_path, label_path,
         test_size, isfusion, fe_load_path, fl_load_path, fn_load_path, fh_load_path, fu_load_path):

    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists('./results/{}'.format(name)):
        os.mkdir('./results/{}'.format(name))


    fN = DAM_Classifier(num_classes).cuda()
    fE = DAMFEncoder(n_channels=3).cuda()
    fL = DAMFDecoder_l(n_channels=num_channels).cuda()
    fH = DAMFDecoder_h(n_channels=num_channels).cuda()
    fU = Fusion()


    fN.load_state_dict(torch.load(fn_load_path))
    fE.load_state_dict(torch.load(fe_load_path))
    fL.load_state_dict(torch.load(fl_load_path))
    fH.load_state_dict(torch.load(fh_load_path))
    fU.load_state_dict(torch.load(fu_load_path))

    fN.eval()
    fE.eval()
    fL.eval()
    fH.eval()
    fU.eval()

    if test_dataset=='nyu':
        test_dataset = NYUUWDataset(data_path,
            label_path,
            size=test_size,
            test_start= 0,
            mode='test')
    else:
        # Add more datasets
        test_dataset = NYUUWDataset(data_path,
            label_path,
            size=100,
            test_start= 0,
            mode='test')

    batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ssim_scores, psnr_scores, mse_scores = test(fE, fL, fN, fH, fU, dataloader, isfusion)

    print ("Average SSIM: {}".format(sum(ssim_scores)/len(ssim_scores)))
    print ("Average PSNR: {}".format(sum(psnr_scores)/len(psnr_scores)))
    print ("Average MSE: {}".format(sum(mse_scores)/len(mse_scores)))

if __name__== "__main__":
    main()