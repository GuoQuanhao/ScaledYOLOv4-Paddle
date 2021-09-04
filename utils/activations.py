import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# Swish https://arxiv.org/pdf/1905.02244.pdf ---------------------------------------------------------------------------
class Swish(nn.Module):  #
    @staticmethod
    def forward(x):
        return x * F.sigmoid(x)


class HardSwish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class MemoryEfficientSwish(nn.Module):
    class F(paddle.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * F.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = F.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(paddle.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(paddle.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = paddle.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return paddle.max(x, self.bn(self.conv(x)))
