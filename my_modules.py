import torch
from torch import nn


class EarlyStopper:

    def __init__(self, metric, patience=10, mode='rel', min_delta=0.01, diff_to=0., verbose=False):

        if mode not in ['rel', 'abs', 'sign', 'dsign', 'signdiffto']:
            raise NotImplementedError('cannot understand early stopping mode: ' + mode)

        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.diff_to = diff_to
        self.verbose = verbose

        self.counter = 0
        self.meta_counter = 0
        self.last_value = float('inf')
        self.second_to_last_value = float('inf')

    def is_converged(self, value):

        if self.mode == 'rel' or self.mode == 'abs':

            delta = abs(value - self.last_value)
            if self.mode == 'rel': delta /= value

            if delta < self.min_delta:
                self.counter += 1
            else:
                self.counter = 0

        elif self.mode == 'sign':

            if value * self.last_value < 0:
                self.counter += 1
            else:
                self.counter = 0

        elif self.mode == 'dsign':

            if (value - self.last_value) * (self.last_value - self.second_to_last_value) < 0:
                self.counter += 1
            else:
                self.counter = 0

        elif self.mode == 'signdiffto':

            if self.verbose > 2:
                print(value)
                print(self.last_value)

            if (value - self.diff_to) * (self.last_value - self.diff_to) < 0:
                self.counter += 1
            else:
                self.counter = 0

        if self.verbose > 1:
            print('\nearly stopping counter: ' + str(self.counter))

        if self.counter >= self.patience:
            if self.verbose > 0:
                print('early stopping criterion fulfilled!')
            return True

        self.second_to_last_value = self.last_value
        self.last_value = value

        return False


class Dummy(nn.Module):

    def __init__(self, n_params):

        super(Dummy, self).__init__()
        self._n_params = n_params

    def forward(self, x):

        return x[:, self._n_params:]


class LogTransform(nn.Module):
    """Returns log-transformed values in range (-inf,inf)
    if x < eps, then x is clamped to eps
    Dorukhan Boncukçu
    """

    def __init__(self, mask, base=None, eps=1e-6):

        super(LogTransform, self).__init__()

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


class LogTransformBack(nn.Module):
    """Dorukhan Boncukçu
    """

    def __init__(self, mask, base=None):

        super(LogTransformBack, self).__init__()
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


class TanhTransform(nn.Module):
    """Returns tanh-transformed values in range
    (-1,0) for x<0
    0 for x=0
    (0,1) for x>0
    with a value of tanh(1)=+-0.76... for x=+-norm
    """

    def __init__(self, mask, norm=1, factor=1):

        super(TanhTransform, self).__init__()

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


class TanhTransformBack(nn.Module):

    def __init__(self, mask, norm=1, factor=1):

        super(TanhTransformBack, self).__init__()

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


class LogitTransform(nn.Module):
    """Returns logit-transformed values in range
    (-c,c) for x in range [eps, 1-eps] (x is clamped to this range)
    with c = logit(1-eps) = 13.8... for eps=1e-6
    with a value of 0 for x=0.5
    """

    def __init__(self, mask, eps=1e-8, tiny=1e-8, factor=1., onnxcompatible=False):

        super(LogitTransform, self).__init__()

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


class LogitTransformBack(nn.Module):
    """
    """

    def __init__(self, mask, factor=1.):

        super(LogitTransformBack, self).__init__()

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


class FisherTransform(nn.Module):
    """
    Applies the Fisher transform to the input data.
    Transforms data using (1/2)*log((1+x)/(1-x))  for x in range (-1, 1).
    This transformation is applied selectively based on a provided mask.
    """

    def __init__(self, mask, eps=1e-6, tiny=1e-6, factor=0.5, onnxcompatible=False):
        super(FisherTransform, self).__init__()

        self._mask = mask
        self.register_buffer('_eps', torch.tensor(eps))
        self.register_buffer('_tiny', torch.tensor(tiny))
        self.register_buffer('_factor', torch.tensor(factor))
        self._onnxcompatible = onnxcompatible

    def forward(self, x):
        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                dim = torch.clamp(dim, min=-1.+self._eps, max=1.-self._eps)  # Ensure x is in the (-1, 1) range
                x_[idim] = self._factor * torch.log((1 + dim) / (1 - dim))
            else:
                x_[idim] = dim

        return torch.t(x_)


class FisherTransformBack(nn.Module):
    """
    Applies the inverse of the Fisher transform to the input data.
    Transforms data back using the inverse function of the Fisher transformation.
    This transformation is applied selectively based on a provided mask.
    """

    def __init__(self, mask, factor=0.5):
        super(FisherTransformBack, self).__init__()

        self._mask = mask
        self.register_buffer('_factor', torch.tensor(factor))

    def forward(self, x):
        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                x_[idim] = (torch.exp( dim / self._factor) - 1) / (torch.exp( dim / self._factor) + 1)
            else:
                x_[idim] = dim

        return torch.t(x_)


class OneHotEncode(nn.Module):
    def __init__(self, source_idx, target_vals, eps=1.):

        super(OneHotEncode, self).__init__()

        self._source_idx = source_idx
        self._target_vals = target_vals
        self.register_buffer('_eps', torch.tensor(eps))

    def forward(self, x):

        onehot = torch.zeros(x.size(dim=0), len(self._target_vals), device=x.device, dtype=x.dtype)
        for ival, val in enumerate(self._target_vals):
            onehot[:, ival:ival+1] = self._eps * (x[:, self._source_idx:self._source_idx+1] == val).int()

        x = torch.cat([
            x[:, :self._source_idx],
            onehot,
            x[:, self._source_idx+1:]
        ], 1)

        return x


class LinearWithSkipConnection(nn.Module):
    def __init__(self, in_features, out_features, n_params, n_vars, skipindices, hidden_features=None, nskiplayers=1, dropout=0, isfirst=False, islast=False, noskipping=False):

        super(LinearWithSkipConnection, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        if hidden_features is None:
            self._hidden_features = max(self._in_features, self._out_features)
        else:
            self._hidden_features = hidden_features
        self._n_params = n_params
        self._n_vars = n_vars
        self._skipindices = skipindices
        self._nskiplayers = nskiplayers
        self._isfirst = isfirst
        self._islast = islast
        self._noskipping = noskipping

        self._linears = nn.ModuleList()
        self._leakyrelus = nn.ModuleList()
        if dropout: self._dropouts = nn.ModuleList()
        else: self._dropouts = None
        for i in range(self._nskiplayers):

            self._linears.append(nn.Linear(in_features=self._in_features if i == 0 else self._hidden_features,
                                           out_features=self._out_features if i == self._nskiplayers-1 else self._hidden_features))

            # nn.init.normal_(self._linears[-1].weight, mean=0.0, std=0.001)
            # nn.init.normal_(self._linears[-1].bias, mean=0.0, std=0.001)

            # nn.init.zeros_(self._linears[-1].weight)
            # nn.init.zeros_(self._linears[-1].bias)

            if i == 0:
                nn.init.kaiming_normal_(self._linears[-1].weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            else:
                nn.init.zeros_(self._linears[-1].weight)
            nn.init.zeros_(self._linears[-1].bias)

            self._leakyrelus.append(nn.LeakyReLU())

            if dropout: self._dropouts.append(nn.Dropout(dropout))

    def forward(self, x):

        if self._in_features < self._out_features:

            if self._isfirst:
                identity = torch.index_select(x, 1, torch.tensor(self._skipindices, device=x.device))
            else:
                identity = torch.index_select(x, 1, torch.tensor([idx for idx in range(len(self._skipindices))], device=x.device))

            identity = torch.cat((identity, torch.zeros((identity.size(dim=0), self._out_features - len(self._skipindices)), device=x.device, dtype=x.dtype)), dim=1)

        elif self._out_features < self._in_features:

            if self._islast and len(self._skipindices) > self._n_vars:
                # if we have skipped also the parameters, we need to drop them in the last layer
                identity = x[:, self._n_params:self._n_params+self._out_features]
            else:
                identity = x[:, :self._out_features]

        else:

            identity = x

        if self._noskipping:
            identity = torch.zeros_like(identity)

        residual = x
        if self._dropouts is not None:
            for ilayer, (linear, leakyrelu, dropout) in enumerate(zip(self._linears, self._leakyrelus, self._dropouts)):
                residual = linear(residual)
                if not (self._islast and ilayer == self._nskiplayers-1):
                    residual = leakyrelu(residual)
                    residual = dropout(residual)
        else:
            for ilayer, (linear, leakyrelu) in enumerate(zip(self._linears, self._leakyrelus)):
                residual = linear(residual)
                if not (self._islast and ilayer == self._nskiplayers-1):
                    residual = leakyrelu(residual)

        return residual + identity


class CastTo16Bit(nn.Module):
    def __init__(self):
        super(CastTo16Bit, self).__init__()

    def forward(self, x):
        return x.half().to(x.dtype)  # don't _really_ cast to 16bit otherwise dtype problems within the model


class DeepJetConstraint(nn.Module):

    def __init__(self, deepjetindices, logittransform=False, logittransformfactor=1., eps=1e-6, tiny=1e-6, nanotransform=False, onnxcompatible=False):

        super(DeepJetConstraint, self).__init__()

        self._deepjetindices = deepjetindices
        self._logittransform = logittransform
        self.register_buffer('_logittransformfactor', torch.tensor(logittransformfactor))
        self._onnxcompatible = onnxcompatible
        self.register_buffer('_eps', torch.tensor(eps))
        self.register_buffer('_tiny', torch.tensor(tiny))
        self._nanotransform = nanotransform

    def forward(self, x):

        thedeepjets = torch.index_select(x, 1, torch.tensor(self._deepjetindices, device=x.device))

        if self._logittransform:
            thedeepjets = torch.sigmoid(thedeepjets / self._logittransformfactor)

        if self._nanotransform:

            B = torch.index_select(thedeepjets, 1, torch.tensor([0], device=x.device))
            CvB = torch.index_select(thedeepjets, 1, torch.tensor([1], device=x.device))
            CvL = torch.index_select(thedeepjets, 1, torch.tensor([2], device=x.device))
            QG = torch.index_select(thedeepjets, 1, torch.tensor([3], device=x.device))

            C = torch.div(B, torch.div(1, CvB) - 1)

            thedeepjets = torch.cat((
                B,  # b
                C,  # c
                torch.mul(1 - QG, (torch.div(C, CvL) - C)),  # l
                torch.mul(QG, (torch.div(C, CvL) - C)),  # g
            ), 1)

        # softmax changes the values, so just normalize like this
        thesum = thedeepjets.sum(dim=1)
        thesum = torch.round(thesum * 1000) / 1000
        normalized = torch.div(thedeepjets, thesum[:, None])

        if self._nanotransform:

            b = torch.index_select(normalized, 1, torch.tensor([0], device=x.device))
            c = torch.index_select(normalized, 1, torch.tensor([1], device=x.device))
            l = torch.index_select(normalized, 1, torch.tensor([2], device=x.device))
            g = torch.index_select(normalized, 1, torch.tensor([3], device=x.device))

            normalized = torch.cat((
                b,  # B
                torch.div(c, c + b),  # CvB
                torch.div(c, c + l + g),  # CvL
                torch.div(g, g + l),  # QG
            ), 1)

        normalized = torch.clamp(normalized, min=self._tiny, max=1.-self._eps)

        if self._logittransform:
            if self._onnxcompatible:
                normalized = self._logittransformfactor * torch.log(normalized/(1-normalized))
            else:
                normalized = self._logittransformfactor * torch.logit(normalized)  # doesn't work with onnx (opset 13)

        if self._deepjetindices[0] > 0:
            x_ = [torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])], device=x.device)),
                  normalized]
            x = torch.cat(x_, dim=1)
        else:
            x = normalized

        return x



class SelectDeepJets(nn.Module):
    """
    """

    def __init__(self, deepjetindices, sigmoidtransform=False, nanotransform=True):

        super(SelectDeepJets, self).__init__()

        self._deepjetindices = deepjetindices
        self._sigmoidtransform = sigmoidtransform
        self._nanotransform = nanotransform

    def forward(self, x):

        thedeepjets = torch.index_select(x, 1, torch.tensor(self._deepjetindices, device=x.device))

        if self._sigmoidtransform:
            thedeepjets = torch.sigmoid(thedeepjets)

        if self._nanotransform:

            B = torch.index_select(thedeepjets, 1, torch.tensor([0], device=x.device))
            CvB = torch.index_select(thedeepjets, 1, torch.tensor([1], device=x.device))
            CvL = torch.index_select(thedeepjets, 1, torch.tensor([2], device=x.device))
            QG = torch.index_select(thedeepjets, 1, torch.tensor([3], device=x.device))

            C = torch.div(B, torch.div(1, CvB) - 1)

            thedeepjets = torch.cat((
                B,  # b
                C,  # c
                torch.mul(1 - QG, (torch.div(C, CvL) - C)),  # l
                torch.mul(QG, (torch.div(C, CvL) - C)),  # g
            ), 1)

        return thedeepjets


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)
        nn.init.kaiming_normal_(self.linear1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(hidden_features, in_features)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x_out = self.linear1(x)
        x_out = self.activation(x_out)
        x_out = self.linear2(x_out)
        x_out = self.activation(x_out)
        return x + x_out


class ResNet(nn.Module):
    def __init__(self, in_features=2, hidden_features=8, out_features=1, n_resblocks=2):
        super(ResNet, self).__init__()

        modules = [
            ('Linear_first', nn.Linear(in_features, hidden_features)),
            ('LeakyReLu_first', nn.LeakyReLU(negative_slope=0.01))
        ]
        for i in range(n_resblocks):
            modules.append(('ResBlock_' + str(i), ResBlock(hidden_features, hidden_features)))
        modules.append(('Linear_last', nn.Linear(hidden_features, out_features)))

        nn.init.kaiming_normal_(modules[0][1].weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(modules[0][1].bias)

        nn.init.zeros_(modules[-1][1].weight)
        nn.init.zeros_(modules[-1][1].bias)

        from collections import OrderedDict
        self.net = nn.Sequential(OrderedDict(modules))

        self.out_features = out_features


    def forward(self, x):
        residual = x
        out = self.net(x)
        return residual[..., -self.out_features:] + out


if __name__ == '__main__':

    # model = ResNet(in_features=6, hidden_features=128, out_features=3, n_resblocks=4)
    # x = torch.rand(2, 6)
    # print(x)
    # print(model(x))

    pass
