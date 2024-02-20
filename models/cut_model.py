import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
from torch.distributions.beta import Beta
from torch.nn import functional as F
from models.PatchGCL import GNNLoss

import torch.nn as nn
from models.VGG import VGG16

import matplotlib.pyplot as plt

class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def get_lambda(alpha=1.0,size=None,device=None):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
#         lam = Beta()
    else:
        lam = 1.
    return lam
def get_spa_lambda(alpha=1.0,size=None,device=None):
    '''Return lambda'''
    if alpha > 0.:
        lam = torch.from_numpy(np.random.beta(alpha, alpha,size=size)).float().to(device)
#         lam = Beta()
    else:
        lam = 1.
    return lam
class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_GNN', type=float, default=1.0, help='weight for HDCE loss: HDCE(G(X), X)')

        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--attn_layers', type=str, default='4,7,9', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--alpha', type=float, default=0.2)


        ##### Graph configs.
        parser.add_argument('--gnn_idt', action='store_true')
        parser.add_argument('--num_hop', type=int, default=2)
        parser.add_argument('--pooling_num', type=int, default=1)
        parser.add_argument('--down_scale', type=int, default=8)
        parser.add_argument('--pooling_ratio', type=str, default='1, 1, 1', help='Ratio for pooling level | [ level0, level1, level2 ]')
        parser.add_argument('--nonzero_th', type=float, default=0.6)

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.train_epoch = None

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']

        if opt.lambda_GNN > 0.0:
            self.loss_names.append('GNN')
            if opt.gnn_idt and self.isTrain:
                self.loss_names += ['GNN_Y']


        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
        self.opt.pooling_ratio = [float(i) for i in self.opt.pooling_ratio.split(',')]
        self.alpha = opt.alpha
        if opt.gnn_idt and self.isTrain:
            self.visual_names += ['idt_B']
        self.normalization = Normalization(self.device)
        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = GNNLoss(opt=opt)
        self.netPre = VGG16().to(self.device)
        
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)

        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

        
        
    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.gnn_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.gnn_idt:
            self.idt_B = self.fake[self.real_A.size(0):]


    def set_epoch(self, epoch):
        self.train_epoch = epoch

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        ## get feat
        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), \
                                                self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B + 1) * 0.5)
        fake_B_feat = self.netPre(norm_fake_B, self.attn_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        real_A_feat = self.netPre(norm_real_A, self.attn_layers, encode_only=True)
        
        self.loss_GNN = self.netF(fake_B_feat, real_A_feat,self.opt.num_patches)
        if self.opt.gnn_idt:
            norm_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            idt_B_feat = self.netPre(norm_idt_B, self.attn_layers, encode_only=True)

            if self.opt.flip_equivariance and self.flipped_for_equivariance:
                idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
            real_B_feat = self.netPre(norm_real_B, self.attn_layers, encode_only=True)

            self.loss_GNN_Y = self.netF(idt_B_feat, real_B_feat,self.opt.num_patches)
        loss_GNN_both = (self.loss_GNN + self.loss_GNN_Y) * 0.5

        self.loss_G = self.loss_G_GAN + self.opt.lambda_GNN*loss_GNN_both 
        return self.loss_G

#--------------------------------------------------------------------------------------------------------

    def init_netF(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization(
            (self.real_B + 1) * 0.5), self.normalization((self.fake_B + 1) * 0.5)

        fake_B_feat = self.netPre(norm_fake_B, self.attn_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        real_A_feat = self.netPre(norm_real_A, self.attn_layers, encode_only=True)

        self.loss_GNN = self.netF(fake_B_feat, real_A_feat, self.opt.num_patches)
