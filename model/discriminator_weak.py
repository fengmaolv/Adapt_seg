import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

                self.drop = nn.Dropout(0.5)
		self.conv = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, padding=0, bias=True)


	def forward(self, x):
                x = self.drop(x)
		x = self.conv(x)
		return x
