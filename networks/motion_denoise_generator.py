import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder, MotionDirection
from einops import repeat

class MDGenerator(nn.Module):
    def __init__(self, image_size=512, style_dim=512, feature_dim=40, motion_dim=40, scale=1, pure_data=False, dictionary=True):
        super(MDGenerator, self).__init__()
        style_dim = style_dim * scale
        self.enc = Encoder(image_size, style_dim, feature_dim, motion_dim, scale, pure_data, dictionary)
        self.dir = MotionDirection(style_dim, feature_dim, motion_dim, pure_data, dictionary)
        self.dec = Decoder(image_size, style_dim, scale)
    def pure_motion_img(self, img_source, imgs_target):
        t, c, h, w = imgs_target.size()
        z_s2r, alpha, feats = self.enc(img_source, imgs_target)
        z_s2r = repeat(z_s2r, 'a b -> (repeat a) b', repeat=t)
        # feats = [repeat(feat, 'b c h w -> (repeat b) c h w', repeat=t) for feat in feats]
        z_s2t = self.dir(z_s2r, alpha)
        imgs_recon = self.dec(z_s2t, feats)
        return imgs_recon, alpha

    def pure_motion_vid(self, vid_target, chunk_size):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        source = vid_target[0:1]
        outputs = []
        for start in range(0, vid_target.size(0), chunk_size):
            output, _ = self.pure_motion_img(source, vid_target[start:start + chunk_size])
            outputs.append(output)
        return torch.cat(outputs, dim=0)
