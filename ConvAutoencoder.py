import torch
import torch.nn as nn
import torch.nn.functional as F

def architecture(inChannels, outChannels, ch=16):
  arch = {}
  arch['in'] = inChannels
  arch['out'] = outChannels
  arch['encoder'] = [
      (ch, 2*ch, True),
      (2*ch, 4*ch, True),
  ]

  arch['decoder'] = [
      (4*ch, 2*ch, True),
      (2*ch, ch, True)
  ]

  arch['channel_ratio'] = 1
  arch['inputChannels'] = ch
  arch['outputChannels'] = ch
  arch['groups'] = 1
  return arch

class Autoencoder(nn.Module):
  def __init__(self, arch):
    super(Autoencoder, self).__init__()

    self.encoderBlocks = nn.ModuleList()
    self.decoderBlocks = nn.ModuleList()
    self.activation = nn.ReLU()

    encArch = arch['encoder']
    decArch = arch['decoder']
    self.inConv = nn.Conv2d(arch['in'], encArch[0][0], 1, padding=0)
    self.outConv = nn.Conv2d(decArch[len(decArch)-1][1], arch['out'], 1, padding=0)

    self.norm = nn.GroupNorm(arch['groups'], decArch[len(decArch)-1][1])

    for block in encArch:
      self.encoderBlocks.append(EncoderBlock(block[0], block[1], pool=block[2], groups=arch['groups'], channel_ratio=arch['channel_ratio']))
    
    for block in decArch:
      self.decoderBlocks.append(DecoderBlock(block[0], block[1], upsample=block[2], groups=arch['groups'], channel_ratio=arch['channel_ratio']))

  def forward(self, x):
    x = self.inConv(x)
    for block in self.encoderBlocks:
      x = block(x)
    
    for block in self.decoderBlocks:
      x = block(x)
    x = self.outConv(self.activation(self.norm(x)))
    #x = F.softmax(x, 1) #pixelwise softmax

    #for now just return raw values
    return x

class EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, channel_ratio=4, pool=True, groups=1):
    super(EncoderBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = in_channels // channel_ratio
    self.residual_channels = out_channels - in_channels
    self.activation = nn.ReLU()
    self.groups=groups

    assert out_channels >= in_channels


    #1x1 Conv
    self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1, padding=0)
    #3x3 Conv 
    self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1)
    #3x3 Conv
    self.conv3 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1)
    #1x1 Conv
    self.conv4 = nn.Conv2d(self.hidden_channels, self.out_channels, 1, padding=0)
    #Average Pool
    if pool:
      self.pool = nn.AvgPool2d(2)
    else:
      self.pool = None
    
    if in_channels != out_channels:
      self.resConv = nn.Conv2d(self.in_channels, self.residual_channels, 1, padding=0)

    self.norm1 = nn.GroupNorm(self.groups, self.in_channels)
    self.norm2 = nn.GroupNorm(self.groups, self.hidden_channels)
    self.norm3 = nn.GroupNorm(self.groups, self.hidden_channels)
    self.norm4 = nn.GroupNorm(self.groups, self.hidden_channels)
        
    #Each block outputs raw un-normalized, un-activated values
  def forward(self, x):
    x = self.activation(self.norm1(x))
    h = self.activation(self.norm2(self.conv1(x)))
    h = self.activation(self.norm3(self.conv2(h)))
    h = self.activation(self.norm4(self.conv3(h)))
    h = self.conv4(h)
    
    if self.in_channels != self.out_channels:
      r = self.resConv(x)
      x = torch.cat([x,r],1) #dimensions should be Example, Channel, Height, Width
    x = h + x
    if self.pool:
      x = self.pool(x)
    return x

class DecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, channel_ratio=4, upsample=True, groups=1):
    super(DecoderBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = in_channels // channel_ratio
    self.activation = nn.ReLU()
    self.groups = groups

    assert out_channels <= in_channels


    #1x1 Conv
    self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1, padding=0)
    #3x3 Conv 
    self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1)
    #3x3 Conv
    self.conv3 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1)
    #1x1 Conv
    self.conv4 = nn.Conv2d(self.hidden_channels, self.out_channels, 1, padding=0)
    #Upsample
    if upsample:
      self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    else:
      self.upsample = None


    self.norm1 = nn.GroupNorm(self.groups, self.in_channels)
    self.norm2 = nn.GroupNorm(self.groups, self.hidden_channels)
    self.norm3 = nn.GroupNorm(self.groups, self.hidden_channels)
    self.norm4 = nn.GroupNorm(self.groups, self.hidden_channels)
        

  def forward(self, x):
    x = self.activation(self.norm1(x))
    h = self.activation(self.norm2(self.conv1(x)))
    h = self.activation(self.norm3(self.conv2(h)))
    h = self.activation(self.norm4(self.conv3(h)))
    h = self.conv4(h)

    x = h + x[:, :self.out_channels]

    if self.upsample is not None:
      x = self.upsample(x)

    return x

