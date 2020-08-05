import torch
import torchvision 
import torch.nn as nn 
import numpy as np
import os
import argparse
import torch.optim as optim
from Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import print_options
from models import encoder3,encoder4
from models import decoder3,decoder4
from models import encoder5 as loss_network
from PIL import Image
from skimage import io, transform
from torchvision import transforms, utils
from attention import Self_Attn as attn
from pytorch_ssim import ssim
from dataset import MPIDataset
from dataset import ImgDataset
from dataset import MonkaaDataset
from dataset import flyingthingsDataset
from dataset import Dataset
from Criterion import LossCriterion
from Criterion import ssim_loss
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                    help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrix_dir", default='',
                    help='pre-trained decoder path')
parser.add_argument("--stylePath", default="",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="",
                    help='path to MPI dataset')
parser.add_argument("--outf", default="trainingOutput/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r31",
                    help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41",
                    help='layers for style')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument("--niter", type=int,default=2000,
                    help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--lr", type=float, default=1e-7,
                    help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0,
                    help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02,
                    help='style loss weight')
parser.add_argument("--ssim_weight", type=float, default=50,
                    help='ssim loss weight')
parser.add_argument("--log_interval", type=int, default=5,
                    help='log interval')
parser.add_argument("--gpu_id", type=int, default=0,
                    help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=1,
                    help='checkpoint save interval')
parser.add_argument("--layer", default="r31",
                    help='which features to transfer, either r31 or r41')
parser.add_argument("--checkpoints_dir", default="",
                    help='the dir saves checkpoints')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
if(opt.cuda):
    torch.cuda.set_device(opt.gpu_id)
cudnn.benchmark = True
print_options(opt)
device = 'cuda'

################# DATA #################
content_dataset = MonkaaDataset(opt.contentPath,opt.loadSize,opt.fineSize)
content_loader = torch.utils.data.DataLoader(dataset     = content_dataset,
                                              batch_size  = opt.batchSize,
                                              shuffle     = True,
                                              num_workers = 1,
                                              drop_last   = True)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                            batch_size  = opt.batchSize,
                                            shuffle     = True,
                                            num_workers = 1,
                                            drop_last   = True)
style_loader = iter(style_loader_)
print('Data load success.')
################# MODEL #################
vgg5 = loss_network()
if(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4()
    dec = decoder4()
attention = attn(256,'relu')
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
vgg5.load_state_dict(torch.load(opt.loss_network_dir))
matrix.load_state_dict(torch.load(opt.matrix_dir))
for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
print('model load success.')
################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,
                          opt.content_layers,
                          opt.style_weight,
                          opt.content_weight,
                          opt.ssim_weight)
ssim_loss = ssim_loss(opt.ssim_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)
################# GLOBAL VARIABLE #################
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    matrix.cuda()
    styleV = styleV.cuda()
    

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

for iteration in range(1,opt.niter+1):

    optimizer.zero_grad()
    try:
        style,_ = style_loader.next()
    except IOError:
        style,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,_ = style_loader.next()
    except:
        continue
    styleV.resize_(style.size()).copy_(style)
    
    for step, (frames,path0) in enumerate(content_loader):
        frames_done = list()
        print("path0",path0)
        ssim_0 = []
        ssim_1 = []
        frames_fea = []
        #l = len(frames)
        l = 10
        for i in range(0,l):
            x_t = frames[i]
            if(i == 0):
                fx_t = vgg(x_t.cuda().squeeze(1))
                #print(type(fx_t))
                fs = vgg(styleV)
                if(opt.layer == 'r41'):
                    feature,transmatrix = matrix(fx_t[opt.layer],fs[opt.layer])
                else:
                    feature,transmatrix = matrix(fx_t,fs)
                frames_fea = feature
                transfer = dec(feature)
                fx_tt = transfer
                frames_done.append(fx_tt)
            else:
                fx_t = vgg(x_t.cuda().squeeze(1))
                fs = vgg(styleV)
                if(opt.layer == 'r41'):
                    feature,transmatrix = matrix(fx_t[opt.layer],fs[opt.layer])
                else:
                    feature,transmatrix = matrix(fx_t,fs)
                
                f1 = feature.cuda()#this frame's feature
                f2 = frames_fea
                out,attention_value = attention(f1,f2)
                #frames_fea.append(out)
                frames_fea = out
                transfer = dec(out)
                #transfer = dec(feature)
                fx_tt = transfer
                frames_done.append(fx_tt)

            content = frames[l-1].cuda()
            sF_loss = vgg5(styleV)
            cF_loss = vgg5(content)
            tF = vgg5(fx_tt)
        #caculate the ssim array of source video and stylized video
        ssim_0 = np.zeros(l-1)
        ssim_1 = np.zeros(l-1)
        m = 5
        ssim_m0 = np.zeros(l-m)
        ssim_m1 = np.zeros(l-m)
        for i in range(0,l-1):
            x_t = frames[i]
            #print(x_t.shape)
            x_t1 = frames[i+1]
            y_t = frames_done[i]
            y_t1 = frames_done[i+1]
            ssim0 = ssim(x_t,x_t1)
            ssim1 = ssim(y_t,y_t1)
            #convert tensor to numpy
            ssim0 = ssim0.cpu()
            ssim1 = ssim1.cpu()
            ssim0 = ssim0.detach().numpy()
            ssim1 = ssim1.detach().numpy()
            ssim_0[i] = ssim0
            ssim_1[i] = ssim1
            ssim_0 = np.array(ssim_0)
            ssim_1 = np.array(ssim_1)
            if (i+m<l):
                x_tm = frames[i+m]
                y_tm = frames_done[i+m]
                ssimm0 = ssim(x_t,x_tm)
                ssimm1 = ssim(y_t,y_tm)
                ssimm0 = ssimm0.cpu()
                ssimm1 = ssimm1.cpu()
                ssimm0 = ssimm0.detach().numpy()
                ssimm1 = ssimm1.detach().numpy()
                ssim_m0[i] = ssimm0
                ssim_m1[i] = ssimm1
                ssim_m0 = np.array(ssim_m0)
                ssim_m1 = np.array(ssim_m1)

        # caculate the loss 
        ssim_0 = torch.from_numpy(ssim_0)
        ssim_1 = torch.from_numpy(ssim_1)
        ssim_m0 = torch.from_numpy(ssim_m0)
        ssim_m1 = torch.from_numpy(ssim_m1)
        ssim_1 = torch.autograd.Variable(ssim_1, requires_grad=True)
        ssim_m1 = torch.autograd.Variable(ssim_m1, requires_grad=True)
        ssim_loss_torch = ssim_loss(ssim_1,ssim_0)
        ssim_loss_torch.float()
        ssim_loss_long_torch = ssim_loss(ssim_m1,ssim_m0)
        ssim_loss_long_torch.float()
        loss,styleLoss,contentLoss,ssim_loss_torch, ssim_loss_long_torch = criterion(tF,sF_loss,cF_loss,ssim_loss_torch.float().cuda(),ssim_loss_long_torch.float().cuda())
        loss.backward(retain_graph=True)
        optimizer.step()

            
        print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f ssim_loss: %.4f ssim_loss_l: %.4f Learng Rate is %.8f'%#
            (opt.niter,iteration,loss,contentLoss,styleLoss,ssim_loss_torch,ssim_loss_long_torch,optimizer.param_groups[0]['lr']))#
            

        adjust_learning_rate(optimizer,iteration)

        if(iteration > 0 and (iteration) % opt.save_interval == 0):
            torch.save(matrix.state_dict(), '%s/%s-604.pth' % (opt.outf,opt.layer))    
