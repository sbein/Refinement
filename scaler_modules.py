import torch
from torch import nn

## Tanh
class TanhScaler(nn.Module):

    def __init__(self, mask, norm=1, factor=1):

        super(TanhScaler, self).__init__()

        self._mask = mask
        self._norm = norm
        self._factor = factor

    def forward(self, x):

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                x_[idim] = self._factor * torch.tanh(dim / self._norm)
            else:
                x_[idim] = dim

        return torch.t(x_)

class TanhInverseScaler(nn.Module):

    def __init__(self, mask, norm=1, factor=1):

        super(TanhInverseScaler, self).__init__()

        self._mask = mask
        self._norm = norm
        self._factor = factor

    def forward(self, x):

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:

                dim = torch.clamp(dim / self._factor, max=1.-1e-7)

                # x_[idim] = 0.5 * torch.log((1 + dim) / (1 - dim)) * self._norm
                x_[idim] = torch.atanh(dim) * self._norm

            else:
                x_[idim] = dim

        return torch.t(x_)

## Log
class LogScaler(nn.Module):

    def __init__(self, mask, base=None, eps=1e-6):

        super(LogScaler, self).__init__()

        self._mask = mask
        if base is None:
            self.register_buffer('_base', torch.tensor(0.))
        else:
            self.register_buffer('_base', torch.tensor(base))
        self.register_buffer('_eps', torch.tensor(eps))

    def forward(self, x):

        self._base = self._base.to(x.device)
        self._eps = self._eps.to(x.device)

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                dim = torch.clamp(dim, min=self._eps)
                x_[idim] = torch.log(dim)
                if self._base > 0.: x_[idim] /= torch.log(self._base)
            else:
                x_[idim] = dim

        return torch.t(x_)

class LogInverseScaler(nn.Module):

    def __init__(self, mask, base=None):

        super(LogInverseScaler, self).__init__()
        self._mask = mask
        if base is None:
            self.register_buffer('_base', torch.tensor(0.))
        else:
            self.register_buffer('_base', torch.tensor(base))

    def forward(self, x):

        self._base = self._base.to(x.device)

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                if self._base > 0.:
                    x_[idim] = torch.pow(self._base, dim)
                else:
                    x_[idim] = torch.exp(dim)
            else:
                x_[idim] = dim

        return torch.t(x_)

# Logit
class LogitScaler(nn.Module):

    def __init__(self, mask, eps=1e-8, tiny=1e-8, factor=1., onnxcompatible=False):

        super(LogitScaler, self).__init__()

        self._mask = mask
        self.register_buffer('_eps', torch.tensor(eps))
        self.register_buffer('_tiny', torch.tensor(tiny))
        self.register_buffer('_factor', torch.tensor(factor))
        self._onnxcompatible = onnxcompatible

    def forward(self, x):

        # somehow this was needed...
        self._tiny = self._tiny.to(x.device)
        self._eps = self._eps.to(x.device)

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                dim = torch.clamp(dim, min=self._tiny, max=1.-self._eps)
                if self._onnxcompatible:
                    x_[idim] = self._factor * torch.log(dim/(1-dim))
                else:
                    x_[idim] = self._factor * torch.logit(dim)  # doesn't work with onnx (opset 13)
            else:
                x_[idim] = dim

        return torch.t(x_)

class LogitInverseScaler(nn.Module):

    def __init__(self, mask, factor=1.):

        super(LogitInverseScaler, self).__init__()

        self._mask = mask
        self.register_buffer('_factor', torch.tensor(factor))

    def forward(self, x):
        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                x_[idim] = torch.sigmoid(dim / self._factor)
            else:
                x_[idim] = dim

        return torch.t(x_)