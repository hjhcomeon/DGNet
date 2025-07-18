import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.conv(weight)
        weight = self.sigmoid(weight)
        return x * weight


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x










class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


# 上采样

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out





class DualResNet(nn.Module):

    # layers（即 [2, 2, 2, 2]）用于指定每个残差层的块数
    def __init__(self, block, layers,num_classes=3, planes=64, spp_planes=128, head_planes=128, augment=True):
        super(DualResNet, self).__init__()
        highres_planes = planes * 2
        self.augment = augment

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        
        
        # CBAM 和 ECA模块的初始化
        self.cbam1 = CBAM(planes)  # 用于layer1的输出
        self.cbam2 = CBAM(planes * 2)  # 用于layer2的输出
        self.cbam3 = CBAM(planes * 4)  # 用于layer3的输出
        self.cbam4 = CBAM(planes * 8)  # 用于layer4的输出

        self.eca1 = ECA(planes)  # 用于layer1的输出
        self.eca2 = ECA(planes * 2)  # 用于layer2的输出
        self.eca3 = ECA(planes * 4)  # 用于layer3的输出
        self.eca4 = ECA(planes * 8)  # 用于layer4的输出
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)            

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        # block：指定残差块的类型（例如 BasicBlock 或 Bottleneck）。
        # inplanes：输入特征图的通道数。
        # planes：输出特征图的通道数。
        # blocks：该层中残差块的数量。
        # stride：卷积操作的步幅，默认值为 1。
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


def forward(self, x):
    # print("x的形状")
    # print(x.shape)
    width_output = x.shape[-1] // 8   # 宽的八分之一
    height_output = x.shape[-2] // 8  # 高的八分之一
    layers = []

    # 进入卷积层1
    x = self.conv1(x)  # 输出112*112，3*3，32，步幅2

    # layer1 后加入 CBAM 和 ECA
    x = self.layer1(x)
    # x = self.cbam1(x)  # CBAM
    # x = self.eca1(x)   # ECA
    layers.append(x)

    # layer2 后加入 CBAM 和 ECA
    x = self.layer2(self.relu(x))
    # x = self.cbam2(x)  # CBAM
    # x = self.eca2(x)   # ECA
    layers.append(x)

    # layer3 后加入 CBAM 和 ECA
    x = self.layer3(self.relu(x))
    # x = self.cbam3(x)  # CBAM
    # x = self.eca3(x)   # ECA
    layers.append(x)

    # layer3_ 后加入 CBAM 和 ECA
    x_ = self.layer3_(self.relu(layers[1]))
    x = x + self.down3(self.relu(x_))
    x_ = x_ + F.interpolate(
                    self.compression3(self.relu(layers[2])),
                    size=[height_output, width_output],
                    mode='bilinear')

    if self.augment:
        temp = x_

    # layer4 后加入 CBAM 和 ECA
    x = self.layer4(self.relu(x))
    # x = self.cbam4(x)  # CBAM
    # x = self.eca4(x)   # ECA
    layers.append(x)

    # layer4_ 后加入 CBAM 和 ECA
    x_ = self.layer4_(self.relu(x_))
    x = x + self.down4(self.relu(x_))
    x_ = x_ + F.interpolate(
                    self.compression4(self.relu(layers[3])),
                    size=[height_output, width_output],
                    mode='bilinear')

    # 最后一层：加入 CBAM 和 ECA
    x_ = self.layer5_(self.relu(x_))
    x = F.interpolate(
                    self.spp(self.layer5(self.relu(x))),
                    size=[height_output, width_output],
                    mode='bilinear')

    x_ = self.final_layer(x + x_)

    if self.augment: 
        x_extra = self.seghead_extra(temp)
        x_ = x_ + x_extra
        return x_
    else:
        return x_



def DualResNet_imagenet(num_classes,pretrained=True):
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, planes=32, spp_planes=128, head_planes=64, augment=True)
    
    if pretrained:
        # ""里面是加载预训练权重，''里面是先加载到cpu，然后转化到gpu
        #################
        # 注释后，模型下载预训练模型，证明这一段代码使用的上的
        pretrained_state = torch.load("model_data/DDRNet23s_imagenet.pth", map_location='cpu') 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        
        model.load_state_dict(model_dict, strict = False)
    return model

def get_seg_model(num_classes,**kwargs):

    model = DualResNet_imagenet(num_classes,pretrained=False)
    return model


if __name__ == '__main__':
    x = torch.rand(3, 800, 800)
    net = DualResNet_imagenet(pretrained=False)
    y = net(x)
    print(y.shape)


















'''




def _make_layer(block, inplanes, planes, blocks, stride=1):
    # block：指定残差块的类型（例如 BasicBlock 或 Bottleneck）。
    # inplanes：输入特征图的通道数。
    # planes：输出特征图的通道数。
    # blocks：该层中残差块的数量。
    # stride：卷积操作的步幅，默认值为 1。
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        if i == (blocks-1):
            layers.append(block(inplanes, planes, stride=1, no_relu=True))
        else:
            layers.append(block(inplanes, planes, stride=1, no_relu=False))

    return nn.Sequential(*layers)


def Sequential1():




def Unet(input_shape=(256,256,3), num_classes=3, backbone = "vgg"):


    planes=64, spp_planes=128, head_planes=128, augment=True
    highres_planes = planes * 2
   
    width_output = input_shape.shape[-1] // 8   # 宽的八分之一
    height_output = input_shape.shape[-2] // 8  # 高的八分之一
    layers = []


    # conv1
    # 第一个卷积层
    x = nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1)(input_shape)
    # 批量归一化
    x = BatchNorm2d(planes, momentum=bn_mom)(x)
    # ReLU 激活函数
    x = nn.ReLU(inplace=True)(x)

    # 第二个卷积层
    x = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1)(x)
    # 批量归一化
    x = BatchNorm2d(planes, momentum=bn_mom)(x)
    # ReLU 激活函数
    x = nn.ReLU(inplace=True)(x)
    
    # layer1
    x = _make_layer(BasicBlock, planes, planes, 2)(x)
    layers.append(x)
    
    # layer2
    x = nn.ReLU(inplace=False)(x)
    x = _make_layer(BasicBlock, planes, planes * 2, 2, stride=2)(x)
    layers.append(x)
    
    # layer3
    x = nn.ReLU(inplace=False)(x)
    x = _make_layer(BasicBlock, planes, planes * 2, 2, stride=2)(x)
    layers.append(x)    
    y = nn.ReLU(inplace=False)(layers[1])
    x_ = _make_layer(BasicBlock, planes * 2, highres_planes, 2)(y)
    
    # layer3_
    
    y = nn.ReLU(inplace=False)(x_)    
    y1 = nn.Sequential(
    nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
    BatchNorm2d(planes * 4, momentum=bn_mom)
                        )(y)
    x = x + y1

    y = F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')
    x_ = x_ + y
    
    if augment:
        temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.final_layer(x + x_)

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_extra, x_]
        else:
            return x_ 


    inputs = Input(input_shape)

    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    feat1 = eca_block(conv1 ,name="eac1")
    up1 =  layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(feat1))

    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    feat2 = eca_block(conv2,name="eac2")
    up2 = layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(feat2))

    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    feat3 = eca_block(conv3,name="eac3")
    pool1 = MaxPooling2D(pool_size=(2, 2))(feat3)
    merge7 = concatenate([feat2, pool1], axis=3)

    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    feat4 = eca_block(conv4,name="eac4")
    pool2 = MaxPooling2D(pool_size=(2, 2))(feat4)
    merge8 = concatenate([feat1, pool2], axis=3)

    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    feat5 = eca_block(conv5,name="eac5")
    merge9 = concatenate([feat1, feat5], axis=3)

    conv9 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    feat6 = eca_block(conv10,name="eac6")
    conv11 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(feat6)

    if backbone == "vgg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(num_classes, 1, activation="softmax")(conv11)

    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
    model = Model(inputs=inputs, outputs=P1)
    return model




























class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 3, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]


        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

 
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
'''