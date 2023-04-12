"""
该代码定义了一个基于ResNet34的Unet网络，用于语义分割任务。主要包括以下几个部分：

expansive_block：扩张块，由两个卷积层、ReLU和BatchNorm组成。用于解码过程中对图像进行上采样和特征融合操作。
final_block：最终块，由一个卷积层、ReLU和BatchNorm组成。用于将解码后的特征图转换为最终的输出图像。
Resnet34_Unet：整个网络的主体部分。首先使用ResNet34作为编码器，对输入图像进行特征提取。然后通过一个卷积层和ReLU，将编码器的输出进行特征扩张。
接下来进行解码操作，使用扩张块和编码器的特征图进行上采样和特征融合，直到得到与原始输入图像大小相同的特征图。
最后通过最终块将特征图转换为输出图像(降channel为Numclasses值)

"""

from .unet_parts import *
#JiangShan，Resnet较为复杂，尽量在原Resnet基础上改动，增加其他模块内容
from MyResnet import ASPP
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
__all__ = ['resnet34']

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

# 定义解码器中的卷积块
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        # 卷积块的结构
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        # 上采样
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out

# 定义最后一层卷积块
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),#padding默认是0，这里为了抵消卷积的消减为1
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block

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
        # self.aspp=ASPP.ASPP(512,[6, 12, 18])#[6, 12, 18]，另一组是[2,4,8]
        #######
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)#size减半
        )
        factor = 2 if bilinear else 1

        ###################################U-net解码层
        self.conv_decode4 = expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        # self.conv_decode0 = expansive_block(64, 32, 32)
        # self.final_layer = final_block(32, n_classes)
        self.final_layer = final_block(64, n_classes)


        self.outc = OutConv(64, n_classes)
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



    def forward(self, x):#这里的x应该是传入的原图（经过dataloder）
        #原版U-net对图像的size四下四上。
        x1 = self.conv1(x)#输入3通道，输出通道=64，图片长宽不变（我改步长为1）
        x1 = self.bn1(x1)#size没变
        x1 = self.relu(x1)#size没变
        x1 = self.maxpool(x1)#size没变（我改步长为1）

        x1 = self.layer1(x1)#输入64，输出通道=64，size不减半，第一个layer比较特别,每个block是输入的两次卷积结果和自身相加，每个layer有多个block。
                            #每个layer的第一个block会完成通道上升和size减半，第一个block进行两次卷积和一次相加
        x2 = self.layer2(x1)#输入64，输出通道=128,图片长宽减半
        x3 = self.layer3(x2)#128，输出通道=256,图片长宽减半
        x4 = self.layer4(x3)#256，输出通道=512,图片长宽减半,
        # print(x4.size())#[batchsize,512,32(h),32(w)]
###############################下面是原U-net网络模型
        # x5 = self.down4(x4)  # 512->1024,图片长宽减半，实际操作是maxpool+两次卷积
        # x = self.up1(x5)#一次转置卷积（size变大两倍），在融合之后（torch.cat），对结果进行两次普通卷积（降维）
        #调用实例，参数传到forward里，这里传入最后一次下采样和倒数第二次下采样的结果，1024维度和512维度
        #试试先上采样，再用残差块？
        #############Jiang
        # x=self.aspp(x4)#空间金字塔池化
        ##############
        # 执行 Bottleneck
        bottleneck = self.bottleneck(x4)#图片size减半
        # 执行 Decode
        decode_block4 = self.conv_decode4(bottleneck,x4)
        decode_block3 = self.conv_decode3(decode_block4, x3)
        decode_block2 = self.conv_decode2(decode_block3, x2)
        decode_block1 = self.conv_decode1(decode_block2, x1)
        # decode_block0 = self.conv_decode0(decode_block1)

        final_layer = self.final_layer(decode_block1)

        return final_layer


def _resnet(arch, n_channels, n_classes,block, layers, pretrained, progress, **kwargs):
    model = ResNet(n_channels, n_classes,block, layers, **kwargs)#在这里输入使用的resnet网络
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)#这里传入block=BasicBlock,layers=[2,2,2,2],

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