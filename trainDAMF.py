import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from PIL import Image
from dataset.dataset import NYUUWDataset
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
from models.networkDAMF import DAM_Classifier, DAMFEncoder, DAMFDecoder_l, DAMFDecoder_h, Fusion
import click
import datetime
from models.loss import *

def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)

    return x
def set_requires_grad(nets, requires_grad=False):
	"""
		Make parameters of the given network not trainable
	"""

	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad

	return requires_grad

def compute_val_metrics(fE, fL, fN, fH, fU, dataloader, fusion):
    """
        Compute SSIM scores for the validation set
    """
    fE.eval()
    fL.eval()
    fN.eval()
    fH.eval()
    fU.eval()

    ssim_scores_l = []
    ssim_scores_h = []
    ssim_scores_u = []
    corr = 0


    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, _ = data
        uw_img = Variable(uw_img).cuda()
        cl_img = Variable(cl_img, requires_grad=False).cuda()

        enc_out, enc_outs = fE(uw_img)
        I, I_c = fN(enc_out)
        fL_out, x_Hinput = fL(enc_out, enc_outs, I_c, fusion)
        fH_out = fH(x_Hinput, enc_outs)
        fU_out = fU(fL_out, fH_out)

        fL_out = to_img(fL_out)
        fH_out = to_img(fH_out)
        gt_h   = to_img(cl_img - fL_out)

        I = F.softmax(I, dim=1)


        if int(I.max(1)[1].item()) == int(water_type.item()):
            corr += 1

        fU_out = (fU_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        fL_out = (fL_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        fH_out = (fH_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        gt_h   = (gt_h   * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

        ssim_scores_l.append(ssim(fL_out, cl_img, multichannel=True))
        ssim_scores_h.append(ssim(fH_out, gt_h, multichannel=True))
        ssim_scores_u.append(ssim(fU_out, cl_img, multichannel=True))

    fE.train()
    fL.train()
    fN.train()
    fH.train()
    fU.train()

    return sum(ssim_scores_l) / len(dataloader), sum(ssim_scores_h) / len(dataloader), \
           sum(ssim_scores_u) / len(dataloader), corr / len(dataloader)


def backward_L_loss(fL, fN, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fL, fusion):
    """
        Backpropagate the reconstruction loss
    """
    I, I_c = fN(enc_out)
    fL_out, _ = fL(enc_out, enc_outs, I_c, fusion)
    fL_out = to_img(fL_out)
    l_l = getsocre(cl_img.clone().detach()*255, fL_out.clone().detach()*255, loss_l=True)
    L_loss = criterion_MSE(fL_out, cl_img) * 30 + color_loss(fL_out, cl_img).requires_grad_(True) * 4 + l_l.requires_grad_(True) * 3
    optimizer_fL.zero_grad()
    L_loss.backward()
    optimizer_fL.step()

    return fL_out, L_loss

def backward_H_loss(fH, fL, fN, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fH,  fusion):
    """
        Backpropagate the reconstruction loss
    """
    I, I_c = fN(enc_out)
    fL_out, fH_input = fL(enc_out, enc_outs, I_c, fusion)
    fH_out = to_img(fH(fH_input, enc_outs))
    gt = to_img(cl_img-fL_out)
    l_h = getsocre(cl_img.clone().detach()*255, gt.clone().detach()*255, loss_l=False)
    H_loss = criterion_MSE(fH_out, cl_img) * 40 + l_h.requires_grad_(True) * 6
    optimizer_fH.zero_grad()
    H_loss.backward()
    optimizer_fH.step()

    return fH_out, fL_out, H_loss

def backward_N_loss(fN, enc_out, actual_target, criterion_CE, optimizer_fN):
    fN_out, _ = fN(enc_out.detach())
    N_loss = criterion_CE(fN_out, actual_target)
    optimizer_fN.zero_grad()
    N_loss.backward()
    optimizer_fN.step()

    return N_loss

def backward_U_loss(fL, fN, fH, fU, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fU, fusion):
    I, I_c = fN(enc_out)
    fL_out, x_Hinput = fL(enc_out, enc_outs, I_c, fusion)
    fH_out = fH(x_Hinput, enc_outs)
    fU_out = fU(fL_out, fH_out)
    fU_out = to_img(fU_out)

    U_loss = criterion_MSE(fU_out, cl_img) * 30
    optimizer_fU.zero_grad()
    U_loss.backward()
    optimizer_fU.step()

    return fL_out, U_loss




def write_to_log(log_file_path, status):
    """
        Write to the log file
    """

    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')

@click.command()
@click.argument('name', default='demo')
@click.option('--data_path', default=None, help='Path of training input data')
@click.option('--label_path', default=None, help='Path of training label data')
@click.option('--learning_rate', default=1e-3, help='Learning rate')
@click.option('--batch_size', default=4, help='Batch size')
@click.option('--start_epoch', default=1, help='Start training from this epoch')
@click.option('--end_epoch', default=200, help='Train till this epoch')
@click.option('--num_classes', default=6, help='Number of water types')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--train_size', default=3000, help='Size of the training dataset')
@click.option('--test_size', default=500, help='Size of the testing dataset')
@click.option('--val_size', default=500, help='Size of the validation dataset')
@click.option('--fe_load_path', default=None, help='Load path for pretrained fE')
@click.option('--fl_load_path', default=None, help='Load path for pretrained fL')
@click.option('--fn_load_path', default=None, help='Load path for pretrained fN')
@click.option('--fh_load_path', default=None, help='Load path for pretrained fH')
@click.option('--fu_load_path', default=None, help='Load path for pretrained fU')
@click.option('--fl_threshold', default=0.9,  help='Train fL till this threshold')
@click.option('--fn_threshold', default=0.85, help='Train fN till this threshold')
@click.option('--fh_threshold', default=0.9,  help='Train fH till this threshold')
@click.option('--isfusion', default=True, help='Continue training from start_epoch')
@click.option('--continue_train', default=True, help='Use adversarial loss during training or not')
def main(name, data_path, label_path, learning_rate, batch_size, start_epoch, end_epoch, num_classes,
         num_channels,
         train_size, test_size, val_size, fe_load_path, fl_load_path, fn_load_path, fh_load_path, fu_load_path,
         fl_threshold, fn_threshold, fh_threshold, continue_train, isfusion):
    fE_load_path = fe_load_path
    fL_load_path = fl_load_path
    fN_load_path = fn_load_path
    fH_load_path = fh_load_path
    fU_load_path = fu_load_path



    # Define datasets and dataloaders
    train_dataset = NYUUWDataset(data_path,
                                 label_path,
                                 size=train_size,
                                 train_start=0,
                                 mode='train')

    val_dataset = NYUUWDataset(data_path,
                               label_path,
                               size=val_size,
                               val_start=0,
                               mode='val')

    test_dataset = NYUUWDataset(data_path,
                                label_path,
                                size=test_size,
                                test_start=0,
                                mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    fN = DAM_Classifier(num_classes).cuda()
    fE = DAMFEncoder(n_channels=3).cuda()
    fL = DAMFDecoder_l(n_channels=num_channels).cuda()
    fH = DAMFDecoder_h(n_channels=num_channels).cuda()
    fU = Fusion().cuda()
    fN_req_grad = False


    criterion_MSE = nn.MSELoss().cuda()
    criterion_CE = nn.CrossEntropyLoss().cuda()

    optimizer_fN = torch.optim.Adam(fN.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_fE = torch.optim.Adam(fE.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_fL = torch.optim.Adam(fL.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_fH = torch.optim.Adam(fH.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_fU = torch.optim.Adam(fU.parameters(), lr=learning_rate, weight_decay=1e-5)



    fN.train()
    fE.train()
    fL.train()
    fH.train()
    fU.train()

    if fE_load_path:
        fE.load_state_dict(torch.load(fE_load_path))
        print('Loaded fE from {}'.format(fE_load_path))
    if fL_load_path:
        fL.load_state_dict(torch.load(fL_load_path))
        print('Loaded fL from {}'.format(fL_load_path))
    if fH_load_path:
        fN.load_state_dict(torch.load(fH_load_path))
        print('Loaded fN from {}'.format(fH_load_path))
    if fU_load_path:
        fN.load_state_dict(torch.load(fU_load_path))
        print('Loaded fN from {}'.format(fU_load_path))
    if fN_load_path:
        fN.load_state_dict(torch.load(fN_load_path))
        print('Loaded fN from {}'.format(fN_load_path))

    if not os.path.exists('./checkpoints/{}'.format(name)):
        os.mkdir('./checkpoints/{}'.format(name))

    log_file_path = './checkpoints/{}/log_file.txt'.format(name)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
    write_to_log(log_file_path, status)

    # Compute the initial cross validation scores
    if continue_train:
        fL_val_ssim, fH_val_ssim, fU_val_ssim, fN_val_acc = compute_val_metrics(fE, fL, fN, fH, fU, val_dataloader, isfusion)
    else:
        fL_val_ssim = -1
        fH_val_ssim = -1
        fU_val_ssim = -1
        fN_val_acc = -1
    fL_val_ssim_best, fU_val_ssim_best = fL_val_ssim, fU_val_ssim



    while fL_val_ssim < fl_threshold and continue_train:
        epoch = start_epoch

        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()

            enc_out, enc_outs = fE(uw_img)
            optimizer_fE.zero_grad()
            optimizer_fE.zero_grad()
            fL_out, I_loss = backward_L_loss(fL, fN, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fL,
                                             fusion=False)

            progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

            optimizer_fE.step()

            if idx % 50 == 0:
                save_image(uw_img.cpu().data, './results/uw_img.png')
                save_image(fL_out.cpu().data, './results/fL_out.png')
                print(progress)
                write_to_log(log_file_path, progress)
            fL_val_ssim, fH_val_ssim, fU_val_ssim, fN_val_acc = compute_val_metrics(fE, fL, fN, fH, fU, val_dataloader,
                                                                                    isfusion)
            status = 'Avg fL val SSIM: {}, Avg fH val SSIM: {},Avg fU val SSIM: {},Avg fN val acc: {}\n'.format(
                fL_val_ssim, fH_val_ssim, fU_val_ssim, fN_val_acc)
            print(status)
            write_to_log(log_file_path, status)

    for epoch in range(start_epoch, start_epoch+end_epoch):
        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()
            actual_target = Variable(water_type, requires_grad=False).cuda()

            enc_out, enc_outs = fE(uw_img)

            if fN_val_acc < fn_threshold:

                if not fN_req_grad:
                    fN_req_grad = set_requires_grad(fN, requires_grad=True)

                N_loss = backward_N_loss(fN, enc_out, actual_target, criterion_CE, optimizer_fN)
                progress = "\tEpoch: {}\tIter: {}\tN_loss: {}".format(epoch, idx, N_loss.item())

            elif fL_val_ssim < fl_threshold:
                if isfusion:
                    if fN_req_grad:
                        fN_req_grad = set_requires_grad(fN, requires_grad=False)

                optimizer_fE.zero_grad()
                fL_out, I_loss = backward_L_loss(fL, fN, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fL, fusion=isfusion)

                progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

                optimizer_fE.step()

                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, './results/uw_img.png')
                    save_image(fL_out.cpu().data, './results/fL_out.png')

            elif fH_val_ssim < fh_threshold:

                fH_out, fL_out, H_loss = backward_H_loss(fH, fL, fN, enc_out, enc_outs, cl_img, criterion_MSE,
                                                 optimizer_fH, isfusion)

                progress = "\tEpoch: {}\tIter: {}\tH_loss: {}".format(epoch, idx, H_loss.item())


                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, './results/uw_img.png')
                    save_image(fH_out.cpu().data, './results/fH_out.png')
                    save_image(fL_out.cpu().data, './results/fL_out.png')

            else:
                fU_out, U_loss = backward_U_loss(fL, fN, fH, fU, enc_out, enc_outs, cl_img, criterion_MSE, optimizer_fU, isfusion)
                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, './results/uw_img.png')
                    save_image(fU_out.cpu().data, './results/fU_out.png')
                progress = "\tEpoch: {}\tIter: {}\tU_loss: {}".format(epoch, idx, U_loss.item())

            if idx % 50 == 0:
                print(progress)
                write_to_log(log_file_path, progress)

        # Save models, fU_val_ssim_best
        if fU_val_ssim_best < fU_val_ssim and fL_val_ssim_best < fL_val_ssim:
            torch.save(fN.state_dict(), './checkpoints/{}/fN.pth'.format(name))
            torch.save(fE.state_dict(), './checkpoints/{}/fE.pth'.format(name))
            torch.save(fL.state_dict(), './checkpoints/{}/fL.pth'.format(name))
            torch.save(fH.state_dict(), './checkpoints/{}/fH.pth'.format(name))
            torch.save(fU.state_dict(), './checkpoints/{}/fU.pth'.format(name))
            status = 'End of epoch. Models saved.'
            print(status)
            write_to_log(log_file_path, status)

        fL_val_ssim, fH_val_ssim, fU_val_ssim, fN_val_acc = compute_val_metrics(fE, fL, fN, fH, fU, val_dataloader, isfusion)
        status = 'Avg fL val SSIM: {}, Avg fH val SSIM: {},Avg fU val SSIM: {},Avg fN val acc: {}\n'.format(
            fL_val_ssim, fH_val_ssim, fU_val_ssim, fN_val_acc)
        print(status)
        write_to_log(log_file_path, status)


if __name__ == "__main__":
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    main()