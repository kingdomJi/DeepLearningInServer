from .unet_parts import *
#JiangShan，Resnet较为复杂，尽量在原Resnet基础上改动，增加上采样模块内容
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
__all__ = ['resnet34','resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):    #核心模块，残差块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):#输入特征图
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)#当x维度与out通道数不相同时，这里的downsample是一次1*1卷积，把x的通道数变成与主枝的输出的通道一致

        out += identity #这里就是卷积后的结果out加上输入，F（x）+x
        out = self.relu(out)

        return out#返回相加后的结果F（x）+x



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#这里先用1*1卷积降维度，再用3*3卷积计算，再用1*1卷积把维度升上去
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels,out_channels, atrous_rates):
        super(ASPP, self).__init__()
        # out_channels = 2048#这里根据输入的特征图的channel更改成和输入的特征图channels一样的数量
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),#叠了5层channel，这里再压成原来channel的大小
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):#x是特征图
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes,block, layers, bilinear=False,replace_stride_with_dilation=None,norm_layer=None,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inplanes = 64
        self.dilation = 1#当使用到空洞卷积（膨胀卷积）时才会使用到的参数，默认为1即普通卷积
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=1, padding=3,  # 输入默认GRB三维，输出64维,这里stride原本是2，我改成1，以免size减半
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)#这里stride原本是2，我给改成1，以免size减半
        self.layer1 = self._make_layer(block, 64, layers[0])#stride用默认1，高宽不减半
        # layers[]存的是生成的block的个数，64是基准通道数，一个_make_layer方法返回一个由多个block组成的小model实例，这里layer1相当于该model的实例对象名称
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # dilate默认为none
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        #####添加空洞空间金字塔池化,加在encoder的结尾
        self.aspp=ASPP(2048,2048,[6, 12, 18])#[6, 12, 18]，另一组是[2,4,8]
        # self.aspp = ASPP(512,512, [6, 12, 18])#resnet34
        #######FPN
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #####
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        ###################################解码层
        # self.up1 = Up_J(1024, 512 // factor, bilinear)#创建实例，参数传init里
        self.up2 = Up_J(512, 256 // factor, bilinear)
        self.up3 = Up_J(256, 128 // factor, bilinear)
        self.up4 = Up_J(128, 64, bilinear)
        self.outc = OutConv(256, n_classes)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):#传入的block是类BasicBlock或。。，planes参数是“基准通道数，blocks是生成的block数量
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,#这里步长为2，会减低2倍特征图尺寸
                            self.base_width, previous_dilation, norm_layer))#以resnet34为例，这里是传入Basicblock的init函数参数
        self.inplanes = planes * block.expansion #
        for _ in range(1, blocks):#生产的block数量取决于blocks参数值大小
            layers.append(block(self.inplanes, planes, groups=self.groups,#这里用默认步长为1，特征图尺寸不变
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers) #返回一个由多个block组成的网络model，该model里的block已经完成初始化

    #########################Jiang,FPN
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape  # b c h w
        # 特征x 2倍上采样(上采样到y的尺寸)后与y相加
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    ###########################


    def forward(self, x):#这里的x应该是传入的原图（经过dataloder）
        # x1 = self.inc(x)#这里是原U-net，64通道
        # x2 = self.down1(x1)#128
        # x3 = self.down2(x2)#256
        # x4 = self.down3(x3)#512
        #原版U-net对图像的size四下四上。
        x1 = self.conv1(x)#输入3通道，输出通道=64，图片长宽不变（我改步长为1）
        x1 = self.bn1(x1)#size没变
        x1 = self.relu(x1)#size没变
        x1 = self.maxpool(x1)#size没变（我改步长为1）

        x1 = self.layer1(x1)#输入64，输出通道=64，size不减半，第一个layer比较特别,每个block是输入的两次卷积结果和自身相加，每个layer有多个block。
                            #每个layer的第一个block会完成通道上升和size减半，第一个block进行两次卷积和一次相加
        x2 = self.layer2(x1)#输入64，输出通道=128,图片长宽减半
        x3 = self.layer3(x2)#128，输出通道=256,图片长宽减半
        x4 = self.layer4(x3)#256，输出通道=512,图片长宽减半
        # print(x4.size())#[batchsize,512,32(h),32(w)]
###############################下面是原U-net网络模型
        # x5 = self.down4(x4)  # 512->1024,图片长宽减半，实际操作是maxpool+两次卷积
        # x = self.up1(x5)#一次转置卷积（size变大两倍），在融合之后（torch.cat），对结果进行两次普通卷积（降维）
        #调用实例，参数传到forward里，这里传入最后一次下采样和倒数第二次下采样的结果，1024维度和512维度
        #试试先上采样，再用残差块？
        # x = self.up2(x4, x3)#传入第一次上采样结果加倒数第三次下采样结果（对称），512和256维度
        # x = self.up3(x, x2)#256->64
        # x = self.up4(x, x1)#64
        #############Jiang
        x4=self.aspp(x4)#空间金字塔池化

        # FPN,自上而下，横向连接
        p5 = self.toplayer(x4)
        p4 = self._upsample_add(p5, self.latlayer1(x3))
        p3 = self._upsample_add(p4, self.latlayer2(x2))
        p2 = self._upsample_add(p3, self.latlayer3(x1))##我需要的融合结果,256channels,H,Wsize与x1相同

        # 平滑处理
        # p5 = p5  # p5直接输出
        # p4 = self.smooth(p4)
        # p3 = self.smooth(p3)
        p2 = self.smooth(p2)

        ##############
        # x = self.up2(x)  # 对Unet而言，传入第一次上采样结果加倒数第三次下采样结果（对称），512和256维度。baseline直接传入512
        # x = self.up3(p2)#256->64
        # x = self.up4(x)#64
        ####################
        logits = self.outc(p2)#256->2
        return logits

class Up_J(nn.Module):#Jiang,改动了U-net的UP，直接上采样，去掉融合连接层.
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):#上采样选择双线性插值或转置卷积
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)#该函数是用来进行转置卷积的，padding默认是0，通道减半，size扩大两倍
            # self.conv = DoubleConv(in_channels, out_channels)#这里是进行两次卷积，两次卷积之后的输出维度是out_channels，这里size不变

    def forward(self, x1):#传入两种维度的特征图，例如1024和512，x1比x2维度高，但长宽窄
        x1 = self.up(x1)#高维的x1进行上采样（转置卷积），降维到和低维的x2一样
        # input is CHW，以下是U-net精髓的与低层次融合部分
        # diffY = x2.size()[2] - x1.size()[2]#H，两特征图高度差
        # diffX = x2.size()[3] - x1.size()[3]#W，两特征图宽度差
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])#对x1进行按顺序左右上下维度填充，把x1的长宽变得和x2一样大
        # # if you have padding issues, see
        # x = torch.cat([x2, x1], dim=1)#经过变换此时输入的x1和x2已经size一样了，这里进行了在class维度上的融合拼接，两512维会拼成一个1024维
        return x1



def _resnet(arch, n_channels, n_classes,block, layers, pretrained, progress, **kwargs):
    model = ResNet(n_channels, n_classes,block, layers, **kwargs)#在这里输入使用的resnet网络
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



#这里是真正被外界调用的接口
def resnet34(n_channels, n_classes,pretrained=False, progress=True, **kwargs):#常用34，50以下比较常用，100以上的太大了
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34',n_channels, n_classes, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(n_channels, n_classes,pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', n_channels, n_classes,Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)