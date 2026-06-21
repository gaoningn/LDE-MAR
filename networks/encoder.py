import math
import torch
from torch import nn
from torch.nn import functional as F
from .ops import (EqualConv2d, EqualLinear, ConvLayer)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, out_channel, 3, norm="gn")
		self.conv2 = ConvLayer(out_channel, out_channel, 3, downsample=True, norm="gn")

		self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False, norm=None)


	def forward(self, x):

		h = x

		h = self.conv1(h)
		h = self.conv2(h)

		skip = self.skip(x)
		h = (h + skip) / math.sqrt(2)

		return h

class EqualMLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int = 2, dropout: float = 0.0,
	             act: str = "gelu", use_layernorm: bool = True):
		super().__init__()
		assert n_hidden >= 1, "n_hidden must be >= 1"
		if act == "gelu":
			activation = nn.GELU()
		elif act == "silu":
			activation = nn.SiLU()
		elif act == "relu":
			activation = nn.ReLU(inplace=True)
		else:
			raise ValueError(f"Unsupported activation: {act}")

		layers = []
		d = in_dim
		for _ in range(n_hidden):
			layers.append(EqualLinear(d, hidden_dim))
			if use_layernorm:
				layers.append(nn.LayerNorm(hidden_dim))
			layers.append(activation)
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			d = hidden_dim
		layers.append(EqualLinear(d, out_dim))
		self.net = nn.Sequential(*layers)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class Encoder2R(nn.Module):
	def __init__(self, img_size=512, latent_dim=512, scale=1):
		super(Encoder2R, self).__init__()

		channels = [int(64*scale), int(128*scale), int(256*scale), int(512*scale)]

		# version1
		self.block1 = ConvLayer(1, channels[0], 1) # 256, 3 -> 64
		self.block2 = nn.Sequential(
			ResBlock(channels[0], channels[1])
		) # 64 -> 128
		self.block3 = nn.Sequential(
			ResBlock(channels[1], channels[2])
		) # 128 -> 256
		self.block4 = nn.Sequential(
			ResBlock(channels[2], channels[3])
		) # 256 -> 512
		self.block5 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512
		self.block6 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512
		self.block7 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512

		self.block_512 = ResBlock(channels[3], channels[3])
		self.block8 = EqualConv2d(channels[3], latent_dim, int(img_size / 128), padding=0, bias=False)

	def forward(self, x):

		res = []
		h = x
		h = self.block1(h) # 256
		res.append(h)
		h = self.block2(h) # 128
		res.append(h)
		h = self.block3(h) # 64
		res.append(h)
		h = self.block4(h) # 32
		res.append(h)
		h = self.block5(h) # 16
		res.append(h)
		h = self.block6(h) # 8
		res.append(h)
		h = self.block7(h) # 4
		res.append(h)
		h = self.block_512(h)
		h = self.block8(h) # 1

		return h.squeeze(-1).squeeze(-1), res[::-1]


class Encoder(nn.Module):
	def __init__(self, img_size=512, dim=512, dim_feature=40, dim_motion=20, scale=1, pure=False, dictionary=True):
		super(Encoder, self).__init__()

		# 2R netmork
		self.enc_2r = Encoder2R(img_size=img_size, latent_dim=dim, scale=scale)

		if not dictionary:
			dim_motion = dim
			dim_feature = dim

		# R2T
		self.a_MLP = EqualMLP(
			in_dim=dim, out_dim=dim_feature,
			hidden_dim=dim, n_hidden=2,
			dropout=0.0, act="gelu",
			use_layernorm=True
		)
		self.b_MLP = EqualMLP(
			in_dim=dim, out_dim=dim_motion,
			hidden_dim=dim, n_hidden=2,
			dropout=0.0, act="gelu",
			use_layernorm=True
		)
		self.feature_P = nn.Parameter(torch.rand(7))
		self.enc_r2n = EqualMLP(
			in_dim=dim, out_dim=dim_motion,
			hidden_dim=dim, n_hidden=2,
			dropout=0.0, act="gelu",
			use_layernorm=True
		)
		self.pure = pure

	def forward(self, input_source, input_target=None):
		if input_target is None:
			z_s2r, feats = self.enc_2r(input_source)
			return z_s2r, None, feats

		z_s2r, feats_s = self.enc_2r(input_source)
		z_t2r, feats_t = self.enc_2r(input_target)
		feature_weights = torch.sigmoid(self.feature_P)
		feats = [
			torch.lerp(target_feat, source_feat, weight)
			for source_feat, target_feat, weight in zip(feats_s, feats_t, feature_weights)
		]

		region_coefficients = F.softmax(self.a_MLP(z_t2r), dim=-1)
		motion_coefficients = torch.tanh(self.b_MLP(z_t2r))
		if self.pure:
			alpha = [region_coefficients, motion_coefficients]
		else:
			noise_coefficients = torch.tanh(self.enc_r2n(z_t2r))
			alpha = [region_coefficients, motion_coefficients, noise_coefficients]
		return z_s2r, alpha, feats
