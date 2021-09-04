# This file contains modules common to various models
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import collections


class Mish(nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (paddle.tanh(F.softplus(x)))
        return x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = paddle.randn([128, 20])
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return input


class Conv(nn.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)
        self.bn = nn.BatchNorm2D(c2)
        self.act = Mish() if act else Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(c1, c_, 1, 1, bias_attr=False)
        self.cv3 = nn.Conv2D(c_, c_, 1, 1, bias_attr=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2D(2 * c_)  # applied to cat(cv2, cv3)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(paddle.concat((y1, y2), axis=1))))


class BottleneckCSP2(nn.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(c_, c_, 1, 1, bias_attr=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2D(2 * c_) 
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(paddle.concat((y1, y2), axis=1))))


class VoVCSP(nn.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoVCSP, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1//2, c_//2, 3, 1)
        self.cv2 = Conv(c_//2, c_//2, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(2, axis=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(paddle.concat((x1,x2), axis=1))


class SPP(nn.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.LayerList([nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(paddle.concat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Layer):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2D(c1, c_, 1, 1, bias_attr=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.LayerList([nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2D(2 * c_) 
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(paddle.concat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(paddle.concat((y1, y2), axis=1))))


class MP(nn.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2D(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Focus(nn.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(paddle.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Layer):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return paddle.concat(x, self.d)


class Flatten(nn.Layer):
    # Use after nn.AdaptiveAvgPool2D(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.reshape([x.shape[0], -1])


class Classify(nn.Layer):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2D(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = paddle.concat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

    



class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias_attr=False):
        super().__init__()
        self.add_sublayer('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_sublayer('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias_attr=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_sublayer('dwconv', nn.Conv2D(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias_attr=bias_attr))
        self.add_sublayer('norm', nn.BatchNorm2D(groups))
    def forward(self, x):
        return super().forward(x)  

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias_attr=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_sublayer('conv', nn.Conv2D(in_channels, out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias_attr=bias_attr))
        self.add_sublayer('norm', nn.BatchNorm2D(out_ch))
        self.add_sublayer('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(Conv(inch, outch, k=3))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.LayerList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = paddle.concat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
        out = paddle.concat(out_, 1)
        return out  
        

class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        
        self.add_sublayer('norm', nn.BatchNorm2D(in_channels))
        self.add_sublayer('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)


class HarDBlock2(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum( self.out_partition[i] )
            real_out_ch = self.out_partition[i][0]
            #print( self.links[i],  self.out_partition[i], accum_out_ch)
            conv_layers_.append( nn.Conv2D(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias_attr=True) )
            bnrelu_layers_.append( BRLayer(real_out_ch) )
            cur_ch = real_out_ch
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += real_out_ch
        #print("Blk out =",self.out_channels)

        self.conv_layers = nn.LayerList(conv_layers_)
        self.bnrelu_layers = nn.LayerList(bnrelu_layers_)
    
    def transform(self, blk, trt=False):
        # Transform weight matrix from a pretrained HarDBlock v1
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k-1][0].weight.shape[0] if k > 0 else 
                       blk.layers[0  ][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias
            
            
            self.conv_layers[i].weight[0:part[0], :, :,:] = w_src[:, 0:in_ch, :,:]
            self.layer_bias.append(b_src)
            
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    #for pytorch, add bias with standalone tensor is more efficient than within conv.bias
                    #this is because the amount of non-zero bias is small, 
                    #but if we use conv.bias, the number of bias will be much larger
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None 

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link) ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chos = sum( self.out_partition[ly][0:part_id] )
                    choe = chos + part[0]
                    chis = sum( link_ch[0:j] )
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :,:,:] = w_src[:, chis:chie,:,:]
            
            #update BatchNorm or remove it if there is no BatchNorm in the v1 block
            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2D):
                self.bnrelu_layers[i] = nn.Sequential(
                         blk.layers[i][1],
                         blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]
                    

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:,0:part[0],:,:] if len(part) > 1 else xout
            #print(i)
            #if self.layer_bias[i] is not None:
            #    xin += self.layer_bias[i].view(1,-1,1,1)

            if len(link) > 1:
                for j in range( len(link) - 1 ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chs = sum( self.out_partition[ly][0:part_id] )
                    che = chs + part[0]                    
                    
                    xin += layers_[ly][:,chs:che,:,:]
                    
            xin = self.bnrelu_layers[i](xin)

            if i%2 == 0 or i == len(self.conv_layers)-1:
                outs_.append(xin)

        out = paddle.concat(outs_, 1)
        return out
    
class ConvSig(nn.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)
        self.act = nn.Sigmoid() if act else Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvSqu(nn.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSqu, self).__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)
        self.act = Mish() if act else Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))
