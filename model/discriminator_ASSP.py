import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		self.conv2d_list = nn.ModuleList()
		for dilation, padding in zip([1,2,3,4], [1,2,3,4]):
			self.conv2d_list.append(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False))
		for m in self.conv2d_list:
			m.weight.data.normal_(0, 0.01)

		self.conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True) 
		self.conv.weight.data.normal_(0, 0.01)       

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


	def forward(self, x):
		x = self.conv2d_list[0](x) + self.conv2d_list[1](x) + self.conv2d_list[2](x) + self.conv2d_list[3](x) 
		x = self.leaky_relu(x)
		x = self.conv(x)
		
		return x
