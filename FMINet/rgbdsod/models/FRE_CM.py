# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rgbdsod.models.encoders.vmamba_freq import Cross_Mamba_Attention_SSM




class FCM(nn.Module):
    def __init__(self, channels):
        super(FCM, self).__init__()

        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)


        self.amp_cma = Cross_Mamba_Attention_SSM(
            d_model=channels,
            ssm_ratio=1,
            d_state=16
        )
        self.pha_cma =  Cross_Mamba_Attention_SSM(
            d_model=channels,
            ssm_ratio=1,
            d_state=16
        )


        self.out1_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.out2_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf, panf):
        B, _, H, W = msf.shape

        msF = torch.fft.rfft2(self.pre1(msf))
        panF = torch.fft.rfft2(self.pre2(panf))


        ms_amp = torch.abs(msF)
        ms_pha = torch.angle(msF)
        pan_amp = torch.abs(panF)
        pan_pha = torch.angle(panF)
        b, _, h, w = ms_amp.shape

        ms_amp = rearrange(ms_amp, "b d h w -> b (h w) d")
        pan_amp = rearrange(pan_amp, "b d h w -> b (h w) d")
        ms_pha = rearrange(ms_pha, "b d h w -> b (h w) d")
        pan_pha = rearrange(pan_pha, "b d h w -> b (h w) d")  # [b,l,d]



        amp_fused1, amp_fused2 = self.amp_cma(ms_amp, pan_amp)
        amp_fused1=amp_fused1.view(b, h, w, -1).permute(0, 3, 1, 2)
        amp_fused2 = amp_fused2.view(b, h, w, -1).permute(0, 3, 1, 2)


        pha_fused1, pha_fused2 = self.pha_cma(ms_pha, pan_pha)
        pha_fused1=pha_fused1.view(b, h, w, -1).permute(0, 3, 1, 2)
        pha_fused2 = pha_fused2.view(b, h, w, -1).permute(0, 3, 1, 2)

        real1 = amp_fused1 * torch.cos(pha_fused1) + 1e-8
        imag1 = amp_fused1 * torch.sin(pha_fused1) + 1e-8
        out1 = torch.complex(real1, imag1) + 1e-8
        out1 = torch.abs(torch.fft.irfft2(out1, s=(H, W), norm='backward'))

        real2 = amp_fused2 * torch.cos(pha_fused2) + 1e-8
        imag2 = amp_fused2 * torch.sin(pha_fused2) + 1e-8
        out2 = torch.complex(real2, imag2) + 1e-8
        out2 = torch.abs(torch.fft.irfft2(out2, s=(H, W), norm='backward'))

        out1=self.out1_conv(out1)
        out2 = self.out2_conv(out2)

        return out1, out2



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms = torch.randn(2, 64, 28, 28).to(device)
    pan = torch.randn(2, 64, 28, 28).to(device)
    model = FCM(channels=64).to(device)
    output = model(ms, pan)
    print(f"Input shape: {ms.shape} -> Output shape: {output[0].shape}")
