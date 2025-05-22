import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from yzyall.yzy.mymodelnew.toolbox.models.segformer.mix_transformer import mit_b4
# from toolbox.models.segformer.mix_transformer import mit_b2
from mydesignmodel.yzy_model.CMSINet.CT_Decoder import CT_decoder
from mydesignmodel.yzy_model.CMSINet.CIR import CIR
from mydesignmodel.yzy_model.CMSINet.CIF import CIF
from mydesignmodel.yzy_model.CMSINet.Last_filter import apply_frequency_filter

class Fusion(nn.Module):
    def __init__(self,num_class=41,embed_dims=[64, 128, 320, 512]):
        super(Fusion,self).__init__()
        # self.add = torch.cat()
        self.channels = [64,128,320,512]
        self.d_mit = mit_b4().cuda()
        self.rgb_mit = mit_b4().cuda()
        # self.rgb_d = mit_b2().cuda()

        self.rde_layer1 = CIR(channel=self.channels[1])  # 128
        self.rde_layer2 = CIR(channel=self.channels[2])  # 320
        self.rde_layer3 = CIR(channel=self.channels[3])  # 512

        self.CIFs = nn.ModuleList([
            CIF(dim=embed_dims[0], reduction=64).cuda(),
            CIF(dim=embed_dims[1], reduction=128).cuda(),
            CIF(dim=embed_dims[2], reduction=320).cuda(),
            CIF(dim=embed_dims[3], reduction=512).cuda()])

        self.conv128_160 = nn.Conv2d(128,160,1,1)
        self.conv320_256 = nn.Conv2d(320,256,1,1)

        self.decoder = CT_decoder()
        self.sig = nn.Sigmoid()

    def forward(self,rgb,dep):
        rgb_list = self.rgb_mit(rgb)
        dep_list = self.d_mit(dep)

        fuse0 = self.CIFs[0](rgb_list[0],dep_list[0])  #64 120 160
        rgb_layer1_rde = self.rde_layer1(fuse0[0], rgb_list[1])
        fuse1 = self.CIFs[1](rgb_layer1_rde,dep_list[1])     #128 60 80

        fuse_r1 = self.conv128_160(fuse1[0])     #320 30 40
        rgb_layer2_rde = self.rde_layer2(fuse_r1, rgb_list[2])
        fuse2 = self.CIFs[2](rgb_layer2_rde,dep_list[2])    #320 30 40

        fuse_r2 = self.conv320_256(fuse2[0])     #512 30 40
        rgb_layer3_rde = self.rde_layer3(fuse_r2, rgb_list[3])
        fuse3 = self.CIFs[3](rgb_layer3_rde,dep_list[3])

        enhanced_feature_map3 = apply_frequency_filter(fuse3[0], filter_type='high_pass', cutoff_freq=0.07)
        enhanced_feature_map3 = fuse3[0] + enhanced_feature_map3

        out,s1,s2 = self.decoder(enhanced_feature_map3, fuse2[0], fuse1[0], fuse0[0])

        fkd1 = fuse0[1]
        fkd2 = fuse1[1]
        fkd3 = fuse2[1]

        return out,s1,s2,fkd1,fkd2,fkd3

    def load_pre_sa(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        # self.rgb_d.load_state_dict(new_state_dict3, strict=False)
        self.d_mit.load_state_dict(new_state_dict3, strict=False)
        self.rgb_mit.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_dmit loading')

if __name__ == '__main__':
    net = Fusion().cuda()
    rgb = torch.randn([2, 3, 480, 640]).cuda()
    d = torch.randn([2, 3, 480, 640]).cuda()
    s = net(rgb, d)
    from mydesignmodel.yzy_model.FindTheBestDec.model.FLOP import CalParams
    CalParams(net, rgb, d)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    print("s.shape:", s.shape)







