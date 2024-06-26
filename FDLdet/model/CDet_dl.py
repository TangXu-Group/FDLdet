import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import model.backbone_resnet as backbone_resnet
from collections import OrderedDict
from torch.nn import init
import math
import matplotlib.pyplot as plt

class PanopticFPN(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone_resnet.__dict__[backbone](pretrained=True)
        self.decoder = FPNDecoder(backbone)
           
    def forward(self, x):
        feats = self.backbone(x)
        outs = self.decoder(feats) 
        
        return outs 
    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, ):
        super(up, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.bilinear = bilinear

    def forward(self, x1, x2):
        if self.bilinear == True:
            x1 = nn.functional.interpolate(x1, scale_factor=2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class FPNDecoder(nn.Module):
    def __init__(self, backbone):
        super(FPNDecoder, self).__init__()
        
        if backbone == 'resnet18' or backbone == 'resnet34':
            print(backbone)
            mfactor = 1
            out_dim = 64
        else:
            mfactor = 4
            out_dim = 128
        
        self.layer0 = nn.Conv2d(64, out_dim, kernel_size=1)
        self.layer1 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1)
        self.layer3 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1)
        self.layer4 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1)
        
        self.up1 = up(out_dim*2, out_dim)
        self.up2 = up(out_dim*2, out_dim)
        self.up3 = up(out_dim*2, out_dim)
        self.up4 = up(out_dim*2, out_dim)
    def forward(self, x):
        
        x0 = self.layer0(x['res0'])
        x1 = self.layer1(x['res1'])
        x2 = self.layer2(x['res2'])
        x3 = self.layer3(x['res3'])
        x4 = self.layer4(x['res4'])
        
        x = {}
        x3 = self.up1(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up3(x2, x1)
        x0 = self.up4(x1, x0)

        return x0#, x1, x2, x3

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 
    
class Classifier(nn.Module):
    def __init__(self, channel_num, class_num):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num*2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel_num*2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channel_num*2, class_num, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

def Load_Weight_FordataParallel(state_dict, need_dataparallel=0):
        if_dataparallel = 1
        for k, v in state_dict.items():
            name = k[:6]
            if name != "module":
                if_dataparallel = 0
        if need_dataparallel == 1:
            if if_dataparallel == 1:
                return state_dict
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = "module."+k 
                    new_state_dict[name] = v 
                return new_state_dict
        else:
            if if_dataparallel == 0:
                return state_dict
            else:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] 
                    new_state_dict[name] = v 
                return new_state_dict     


class ChangeDetector(nn.Module):
    def __init__(self, channel_num, class_num, key, word_num,backbone='resnet50'):
        super(ChangeDetector, self).__init__()
        self.feature_extractor_deep = PanopticFPN(backbone=backbone)

        self.classifier = Classifier(word_num*2, class_num)
        
        self.convsf1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_num),
        )
        self.convsf2 = nn.Sequential(
            nn.Conv2d(channel_num, word_num, kernel_size=1, bias=False),
            nn.BatchNorm2d(word_num),
        )
        self.image_conv = double_conv(3,word_num)
        
        word_length = word_num
        self.dictionary = nn.Parameter(init.kaiming_uniform_(torch.randn(1,1,word_num,word_length), a=math.sqrt(5))[0,0])
        
        self.fcd1 = nn.Linear(word_num, channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.fcd2 = nn.Linear(channel_num, channel_num)
        
        self.eye = nn.Parameter(torch.eye(word_num,word_num))
        self.eye.requires_grad = False
        self.key = key
        self.bn1 = nn.BatchNorm2d(word_num)
        self.bn2 = nn.BatchNorm2d(word_num)
        
        self.upsample = nn.UpsamplingBilinear2d(size=(256,256))
        self.channel_num = channel_num
        
        self.conv_t = nn.Conv2d(word_num*2, word_num, kernel_size=1, bias=False)
        
    def Feature_meanlize(self, f, seg, num, max_num):

        bs, f_c, f_w, f_h = f.shape[0],f.shape[1],f.shape[2],f.shape[3]
        single_long = f_w*f_h
        single_base = (torch.arange(bs)*single_long).view(bs,1,1).cuda()

        seg_ = seg[:,:torch.max(num),:torch.max(max_num)]

        seg_onehot = (seg_>0).float().unsqueeze(3)

        leng = torch.sum(seg_>0,dim=2)
        seg_ = (seg_+single_base).reshape(-1)

        f_ = f.permute(0,2,3,1).reshape(bs*f_w*f_h,-1)
        f_x = f_[seg_.long()].reshape(bs,torch.max(num),torch.max(max_num),f_c)
        f_x = f_x * seg_onehot

        f_x = torch.sum(f_x, dim=2)/(leng.unsqueeze(2)+(leng.unsqueeze(2)==0).float())
        
        f_x = f_x.repeat(torch.max(max_num),1,1,1).permute(1,2,0,3).reshape(bs*torch.max(num)*torch.max(max_num),f_c)

        f_[seg_.long()] = f_x
        f_ = f_.reshape(bs,f_w,f_h,f_c).permute(0,3,1,2)
        
        return f_
    
    def dic_learning(self, f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2):
        torch.cuda.empty_cache()
        S = torch.sigmoid(self.convsf2(self.convsf1(f+m)))
        S1 = S[:batch_size]
        S2 = S[batch_size:]
        S1 = self.Feature_meanlize(S1.detach(), seg1, num1, max_num1)+S1
        S2 = self.Feature_meanlize(S2.detach(), seg2, num2, max_num2)+S2     
        Word_set1 = self.bn1(torch.matmul(S1.permute(0,2,3,1), self.dictionary).permute(0,3,1,2))
        Word_set2 = self.bn2(torch.matmul(S2.permute(0,2,3,1), self.dictionary).permute(0,3,1,2))
        Word_set1 = self.upsample(Word_set1)
        Word_set2 = self.upsample(Word_set2)
        c = torch.cat((Word_set1,Word_set2),dim=1)
        torch.cuda.empty_cache()
        return c
    
    def forward(self, i1, i2, seg1, num1, max_num1, seg2, num2, max_num2):
        
        batch_size = i1.shape[0]
        i = torch.cat((i1,i2),dim=0)
        f0 = self.feature_extractor_deep(i) #, f1, f2, f3

        m = self.relu1(self.fcd1(torch.mean(self.dictionary,dim=1).unsqueeze(0)))
        m = self.fcd2(m).reshape(1, self.channel_num, 1,1)
        c0 = self.dic_learning(f0, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2)
        c = self.classifier(c0)
        
        return c
