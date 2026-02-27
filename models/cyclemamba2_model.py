import torch
import itertools
from .base_model import BaseModel
from . import networks
import kornia
import warnings
warnings.filterwarnings('ignore')
from .networks import SemanticConsistencyLossCalculator


class CycleMamba2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned',direction= 'AtoB')       
        if is_train:
            parser.set_defaults(gan_mode='vanilla', pool_size=0)
            parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Se_A', type=float, default=100.0, help='weight for sematic loss, visible')
            parser.add_argument('--lambda_Se_B', type=float, default=100.0, help='weight for sematic loss, infrared')
            parser.add_argument('--lambda_ssim_A', type=float, default=100.0, help='weight for ssim loss, visible')
            parser.add_argument('--lambda_ssim_B', type=float, default=100.0, help='weight for ssim loss, infrared')
            parser.add_argument('--lambda_cycle', type=float, default=100.0, help='weight for cycle loss, rec and visible')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN','G_L1', 'D_real', 'D_fake','Semantic','SSIM','cycle_A','All']# 
        visual_names_B = ['real_B','fake_B']
        visual_names_A = ['real_A','rec_A']
        self.loss_and_weights =[        
            'self.loss_G= self.loss_G_GAN + self.loss_G_L1+self.loss_cycle_A + self.loss_Semantic+self.loss_SSIM ',    #
            'self.loss_G_GAN = self.criterionGAN(self.netD(fake_AB), True)',
            "self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) *self.opt.lambda_L1    #  100",  
            'self.loss_SSIM_A =  self.SSIM(self.real_A, self.rec_A)* self.opt.lambda_ssim_A     # 100 ',   
            "self.loss_SSIM_B = self.SSIM(self.fake_B, self.real_B)*self.opt.lambda_ssim_B      # 100" ,
            'self.loss_SSIM = self.loss_SSIM_A+self.loss_SSIM_B',
            'self.loss_Semantic_A = self.Semantic.calculate_loss(self.real_A, self.rec_A) * self.opt.lambda_Se_A    #100 ',
            'self.loss_Semantic_B = self.Semantic.calculate_loss(self.fake_B, self.real_B) * self.opt.lambda_Se_B   #100',
            'self.loss_Semantic = self.loss_Semantic_A+self.loss_Semantic_B',
            'self.loss_cycle_A = self.criterionL1(self.rec_A, self.real_A) * self.opt.lambda_cycle  # 100',
        ]
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D']
        else:  # during test time, only load G
            self.model_names = ['G_A']
        # define networks (both generator and discriminator)   

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.load_Bnetworks(name = 'G_B', epoch=opt.pre_epoch)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD= networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            # self.gradient = kornia.filters.SpatialGradient()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss() 
            self.SSIM = kornia.losses.SSIMLoss(11, reduction='mean')
            self.Semantic = SemanticConsistencyLossCalculator()
            self.loss_All = 0
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)    
        with torch.no_grad():
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))  


    def backward_D(self):
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss and calculate gradients
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()
        return self.loss_D
    

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_GAN = self.criterionGAN(self.netD(fake_AB), True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) *self.opt.lambda_L1    
        self.loss_SSIM_A =  self.SSIM(self.real_A, self.rec_A)* self.opt.lambda_ssim_A             # On the AVIID-1 is set to100，AVIID-2,AVIID-3 set to 10
        self.loss_SSIM_B = self.SSIM(self.fake_B, self.real_B)*self.opt.lambda_ssim_B
        self.loss_SSIM = self.loss_SSIM_A+self.loss_SSIM_B


        self.loss_Semantic_A = self.Semantic.calculate_loss(self.real_A, self.rec_A)* self.opt.lambda_Se_A   #
        self.loss_Semantic_B = self.Semantic.calculate_loss(self.fake_B, self.real_B) * self.opt.lambda_Se_B   
        self.loss_Semantic = self.loss_Semantic_A+self.loss_Semantic_B


        self.loss_cycle_A = self.criterionL1(self.real_A, self.rec_A) * self.opt.lambda_cycle 

        self.loss_G= self.loss_G_GAN + self.loss_G_L1+self.loss_cycle_A   +self.loss_SSIM + self.loss_Semantic 


        self.loss_All = self.loss_G
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   
        # update D 
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G      更新G时首先D不需要梯度，然后将G梯度设置为0，反向传播，更新权重
        self.set_requires_grad(self.netG_B, False)
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights        
