import os
import torch.nn as nn
nc=2#pred_mask的channel是2
ndf=64
# Number of GPUs available. Use 0 for CPU mode.（可用的GPU数。CPU模式使用0。）
ngpu = 1
# 权重初始化函数，为生成器和判别器模型初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class Discriminator_Upsample(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_Upsample, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x 448 x 448
        nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),#16,in_channels=nc=3,out_channels=ndf=16,kernel_size,stride,padding
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 224x 224
        # nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),#32
        # nn.BatchNorm2d(ndf * 2),
        # nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 112 x112
        nn.Conv2d(ndf , ndf * 4,3, 2, 1, bias=False),#64
        # nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 56 x 56
        nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),#128
        # nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 28 x 28
        nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False),  # 256
        # nn.BatchNorm2d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*16) x 14 x 14
        nn.Conv2d(ndf * 16, 1, 3, 2, 1, bias=False),  # 512
        # nn.BatchNorm2d(ndf * 32),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*32) x 7 x 7
        # nn.Conv2d(ndf * 32, ndf * 64, 3, 2, 1, bias=False),#1024
        ##state size. (ndf*64) x 4 x 4
        # nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False),  # 1*1
        #1*1*1
        nn.Upsample(scale_factor=32, mode='bilinear'),
        nn.Sigmoid()
        )

    def forward(self, input):#输入decoder解码的特征图
        return self.main(input)

class Discriminator_448(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_448, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x 448 x 448
        nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),#16,in_channels=nc=3,out_channels=ndf=16,kernel_size,stride,padding
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 224x 224
        nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),#32
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 112 x112
        nn.Conv2d(ndf*2 , ndf * 4,3, 2, 1, bias=False),#64
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 56 x 56
        nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),#128
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 28 x 28
        nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False),  # 256
        nn.BatchNorm2d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*16) x 14 x 14
        nn.Conv2d(ndf * 16,ndf * 32, 3, 2, 1, bias=False),  # 512
        nn.BatchNorm2d(ndf * 32),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*32) x 7 x 7
        nn.Conv2d(ndf * 32, ndf * 64, 3, 2, 1, bias=False),#1024
        nn.BatchNorm2d(ndf * 64),
        nn.LeakyReLU(0.2, inplace=True),
        ##state size. (ndf*64) x 4 x 4
        nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False),  # 1*1
        nn.Sigmoid()
        )

    def forward(self, input):#输入decoder解码的特征图
        return self.main(input)
class Discriminator_256(nn.Module):
    def __init__(self, ngpu,ndf=16):
        super(Discriminator_256, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x 64 x 256
        nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),#in_channels=nc=3,out_channels=ndf=64,kernel_size,stride,padding
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 128
        nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),#129
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 64
        nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),#256
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 32
        nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),#512
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 16
        nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False),#
        nn.BatchNorm2d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),
        ##8
        nn.Conv2d(ndf * 16, ndf * 32, 3, 2, 1, bias=False),  #
        nn.BatchNorm2d(ndf * 32),
        nn.LeakyReLU(0.2, inplace=True),
        #4
        nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),  #1
        nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):#原版
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),#in_channels=nc=3,out_channels=ndf=64,kernel_size,stride,padding
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),#129
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),#256
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),#512
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),#
        nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class FCDiscriminator(nn.Module):#全连接层辨别器
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)#尺寸减半
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)
		return x#输出尺寸是原尺寸的1/32
# Create the Discriminator
# netD = Discriminator(ngpu).to(device)
# netG = #生成器替换成TransUnet
# # Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))
# # Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
# # Apply the weights_init function to randomly initialize all weights
# #  to mean=0, stdev=0.2.
# # netG.apply(weights_init)
# netG.load_state_dict(torch.load(args.load, map_location=device))#G加载预训练模型
# netD.apply(weights_init)
#
#
# # Establish convention for real and fake labels during training
# real_label = 1.0
# fake_label = 0.0
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))#这个是鉴别网络的优化器，尽可能鉴别准
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))#这个就是我的主干网络优化器，尽可能让在两个数据集上的预测结果让鉴别器挑不出毛病
#
# for epoch in range(num_epochs):
#     import time
#     start = time.time()
#  # For each batch in the dataloader
#     for i, data in enumerate(dataloader, 0):
#      ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#     ###########################
#         ## Train with all-real batch
#         netD.zero_grad()
#      # Format batch
#         real_cpu = data[0].to(device)#real_cpu，公共裂缝的预测图（通过预训练模型得到的正确的预测图）
#         b_size = real_cpu.size(0)
#         label = torch.full((b_size,), real_label, device=device)#给label全填1，造一个全对的label，size和real_cpu对应
#      # Forward pass real batch through D
#         print(real_cpu.size())#(8,3,256,256)/(64,3,64,64)
#      # netD需要有两个输入，一个是待迁移的特征图，一个是迁移的特征图（都是decoder的解码层输出）
#         output = netD(real_cpu).view(-1)#这里输入的是公共裂缝的预测图，real_cpu.size=
#      # Calculate loss on all-real batch
#         print(output.size())#1352=13*13*8/64
#         print(label.size())#8/64
#         errD_real = criterion(output, label)#计算损失
#      # Calculate gradients for D in backward pass
#         errD_real.backward()#反向传播
#         D_x = output.mean().item()
#
#      ## Train with all-fake batch
#             # Generate batch of latent vectors
#         noise = torch.randn(b_size, nz, 1, 1, device=device)
#      # Generate fake image batch with G
#         fake = netG(noise)#noise为地裂缝的预测图
#         label.fill_(fake_label)#label填充0，全错
#      # Classify all fake batch with D
#         output = netD(fake.detach()).view(-1)#这里输入地裂缝的预测图（decoder解码结果），判别地裂缝预测图的情况
#      # Calculate D's loss on the all-fake batch
#         errD_fake = criterion(output, label)#对地裂缝的判别结果与全假之间的差距
#      # Calculate the gradients for this batch
#         errD_fake.backward()#辨别器要尽可能减少地裂缝与假之间的距离，增加判断准确性
#         D_G_z1 = output.mean().item()
#      # Add the gradients from the all-real and all-fake batches
#         errD = (errD_real + errD_fake)/2#计算总损失
#      # Update D
#         optimizerD.step()
#
#      ############################
#         # (2) Update G network: maximize log(D(G(z)))
#     ###########################
#         netG.zero_grad()
#         label.fill_(real_label)  # fake labels are real for generator cost
#             # Since we just updated D, perform another forward pass of all-fake batch through D
#         output = netD(fake).view(-1)
#      # Calculate G's loss based on this output
#         errG = criterion(output, label)#地裂缝预测图与全真的差距
#      # Calculate gradients for G
#         errG.backward()#反向传播，减小地裂缝预测图与全真的差距（迷惑判别器）
#         D_G_z2 = output.mean().item()
#      # Update G
#         optimizerG.step()



