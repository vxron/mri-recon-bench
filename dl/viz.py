from fastmri.models.unet import Unet
import torch

# see what specific unet struct is in fastmri so we can freeze everything up to last 1-2 up-conv blocks...
model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
for name, param in model.named_parameters():
    print(name, param.shape)

"""
Encoder:
  down_sample_layers.0  (32 ch)
  down_sample_layers.1  (64 ch)
  down_sample_layers.2  (128 ch)
  down_sample_layers.3  (256 ch)
  conv                  (512 ch)  <- bottleneck

Decoder:
  up_transpose_conv.0 + up_conv.0   (256 ch)
  up_transpose_conv.1 + up_conv.1   (128 ch)
  up_transpose_conv.2 + up_conv.2   (64 ch)
  up_transpose_conv.3 + up_conv.3   (32 ch -> 1 ch)  <- output block
"""
