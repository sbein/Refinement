import torch
from torch import nn

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