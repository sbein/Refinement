import torch
from torch import nn
from typing import Optional, List


class LogTransform(nn.Module):
    """Returns log10-transformed values in range
    <0 for x<min
    0 for x=min
    1 for x=max
    >1 for x>max
    """

    def __init__(self, min=1., max=None):

        super(LogTransform, self).__init__()

        self._min = torch.tensor(min)
        self._max = torch.tensor(max)

    def forward(self, x):
        if self._max:
            return (torch.log10(x) - torch.log10(self._min)) / (torch.log10(self._max) - torch.log10(self._min))
        else:
            return torch.log10(x) - torch.log10(self._min)

    # def back(self, x):
    #     if self._max:
    #         return torch.pow(10., x * (torch.log10(self._max) - torch.log10(self._min)) + torch.log10(self._min))
    #     else:
    #         return torch.pow(10., x + torch.log10(self._min))


class TanhTransform(nn.Module):
    """Returns tanh-transformed values in range
    (-1,0) for x<0
    0 for x=0
    (0,1) for x>0
    with a value of tanh(1)=+-0.76... for x=+-norm
    """

    def __init__(self, mask, norm=1):

        super(TanhTransform, self).__init__()

        self._mask = mask
        self._norm = norm

    def forward(self, x):

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                x_[idim] = torch.tanh(dim / self._norm)
            else:
                x_[idim] = dim

        return torch.t(x_)

        # return torch.tanh(x / self._norm)

    # def back(self, x):
    #     return torch.atanh(x) * self._norm


class TanhTransformBack(nn.Module):
    """Returns tanh-transformed values in range
    (-1,0) for x<0
    0 for x=0
    (0,1) for x>0
    with a value of tanh(1)=+-0.76... for x=+-norm
    """

    def __init__(self, mask, norm=1):

        super(TanhTransformBack, self).__init__()

        self._mask = mask
        self._norm = norm

    def forward(self, x):

        if self.training:

            return x

        else:

            xt = torch.t(x)
            x_ = torch.empty_like(xt)
            for idim, dim in enumerate(xt):
                if self._mask[idim]:

                    x_[idim] = 0.5 * torch.log((1 + dim) / (1 - dim)) * self._norm
                    # x_[idim] = torch.atanh(dim) * self._norm
                else:
                    x_[idim] = dim

            return torch.t(x_)

            # return torch.atanh(x) * self._norm


class LogitTransform(nn.Module):
    """Returns logit-transformed values in range
    (-c,c) for x in range [eps, 1-eps] (x is clamped to this range)
    with c = logit(1-eps) = 13.8... for eps=1e-6
    with a value of 0 for x=0.5
    """

    def __init__(self, mask, eps=1e-6, factor=1., onnxcompatible=False):

        super(LogitTransform, self).__init__()

        self._mask = mask
        self.register_buffer('_eps', torch.tensor(eps))
        self.register_buffer('_factor', torch.tensor(factor))
        self._onnxcompatible = onnxcompatible

    def forward(self, x):

        # self._eps.to(x.device)

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:

                if self._onnxcompatible:

                    dim[dim < self._eps] = self._eps
                    dim[dim > 1-self._eps] = 1-self._eps

                    x_[idim] = self._factor * torch.log(dim/(1-dim))

                else:

                    x_[idim] = self._factor * torch.logit(dim, eps=self._eps)  # doesn't work with onnx (opset 13)

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

        # if self.training:
        #
        #     return x
        #
        # else:

        # TODO: really??

        xt = torch.t(x)
        x_ = torch.empty_like(xt)
        for idim, dim in enumerate(xt):
            if self._mask[idim]:
                x_[idim] = torch.sigmoid(dim / self._factor)
            else:
                x_[idim] = dim

        return torch.t(x_)

            # return torch.sigmoid(x)


class DeepJetTransform5to4(nn.Module):
    """
    """

    def __init__(self, deepjetindices):

        super(DeepJetTransform5to4, self).__init__()

        self._deepjetindices = deepjetindices

    def forward(self, x):

        B = torch.index_select(x, 1, torch.tensor(self._deepjetindices[0], device=x.device))
        C = torch.index_select(x, 1, torch.tensor(self._deepjetindices[1], device=x.device))
        CvL = torch.index_select(x, 1, torch.tensor(self._deepjetindices[3], device=x.device))
        QG = torch.index_select(x, 1, torch.tensor(self._deepjetindices[4], device=x.device))

        analytical = torch.cat((
            B,  # b
            C,  # c
            torch.mul(1 - QG, (torch.div(C, CvL) - C)),  # l
            torch.mul(QG, (torch.div(C, CvL) - C)),  # g
        ), 1)

        if self._deepjetindices[0] > 0:
            x_ = [torch.index_select(x, 1,
                                     torch.tensor([idx for idx in range(self._deepjetindices[0])], device=x.device)),
                  analytical]
            x = torch.cat(x_, dim=1)
        else:
            x = analytical

        return x


class DeepJetTransform4to5(nn.Module):
    """
    """

    def __init__(self, deepjetindices):

        super(DeepJetTransform4to5, self).__init__()

        self._deepjetindices = deepjetindices

    def forward(self, x):

        if self.training:

            return x

        else:

            b = torch.index_select(x, 1, torch.tensor(self._deepjetindices[0], device=x.device))
            c = torch.index_select(x, 1, torch.tensor(self._deepjetindices[1], device=x.device))
            l = torch.index_select(x, 1, torch.tensor(self._deepjetindices[2], device=x.device))
            g = torch.index_select(x, 1, torch.tensor(self._deepjetindices[3], device=x.device))

            analytical = torch.cat((
                b,  # B
                c,  # C
                torch.div(c, c + b),  # CvB
                torch.div(c, c + l + g),  # CvL
                torch.div(g, g + l),  # QG
            ), 1)

            if self._deepjetindices[0] > 0:
                x_ = [torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])],
                                                            device=x.device)),
                      analytical]
                x = torch.cat(x_, dim=1)
            else:
                x = analytical

            return x


class DeepJetTransform4to4fromNano(nn.Module):
    """
    """

    def __init__(self, deepjetindices):

        super(DeepJetTransform4to4fromNano, self).__init__()

        self._deepjetindices = deepjetindices

    def forward(self, x):

        B = torch.index_select(x, 1, torch.tensor(self._deepjetindices[0], device=x.device))
        CvB = torch.index_select(x, 1, torch.tensor(self._deepjetindices[1], device=x.device))
        CvL = torch.index_select(x, 1, torch.tensor(self._deepjetindices[2], device=x.device))
        QG = torch.index_select(x, 1, torch.tensor(self._deepjetindices[3], device=x.device))

        C = torch.div(B, torch.div(1, CvB) - 1)

        analytical = torch.cat((
            B,  # b
            C,  # c
            torch.mul(1 - QG, (torch.div(C, CvL) - C)),  # l
            torch.mul(QG, (torch.div(C, CvL) - C)),  # g
        ), 1)

        if self._deepjetindices[0] > 0:

            # x_ = [torch.index_select(x, 1,
            #                          torch.tensor([idx for idx in range(self._deepjetindices[0])], device=x.device)),
            #       analytical]
            # x = torch.cat(x_, dim=1)

            x = torch.cat([
                x[:, :self._deepjetindices[0]],
                analytical
            ], dim=1)

        else:
            x = analytical

        return x


class DeepJetTransform4to4toNano(nn.Module):
    """
    """

    def __init__(self, deepjetindices):

        super(DeepJetTransform4to4toNano, self).__init__()

        self._deepjetindices = deepjetindices

    def forward(self, x):

        if self.training:

            return x

        else:

            b = torch.index_select(x, 1, torch.tensor(self._deepjetindices[0], device=x.device))
            c = torch.index_select(x, 1, torch.tensor(self._deepjetindices[1], device=x.device))
            l = torch.index_select(x, 1, torch.tensor(self._deepjetindices[2], device=x.device))
            g = torch.index_select(x, 1, torch.tensor(self._deepjetindices[3], device=x.device))

            analytical = torch.cat((
                b,  # B
                torch.div(c, c + b),  # CvB
                torch.div(c, c + l + g),  # CvL
                torch.div(g, g + l),  # QG
            ), 1)

            if self._deepjetindices[0] > 0:

                # x_ = [torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])],
                #                                             device=x.device)),
                #       analytical]
                # x = torch.cat(x_, dim=1)

                x = torch.cat([
                    x[:, :self._deepjetindices[0]],
                    analytical
                ], dim=1)

            else:
                x = analytical

            return x


class DeepJetConstraint4(nn.Module):

    def __init__(self, deepjetindices, applylogittransform=False, logittransformfactor=1., skipone=False, onnxcompatible=False, eps=1e-6, nanotransform=False):

        super(DeepJetConstraint4, self).__init__()

        self._deepjetindices = deepjetindices
        self._applylogittransform = applylogittransform
        self.register_buffer('_logittransformfactor', torch.tensor(logittransformfactor))
        self._skipone = skipone
        self._onnxcompatible = onnxcompatible
        self.register_buffer('_eps', torch.tensor(eps))
        self._nanotransform = nanotransform

    def forward(self, x):

        # skip last index because 5 DeepJet values have been transformed to 4
        # sumexp = torch.exp(torch.index_select(x, 1, torch.tensor(self._deepjetindices[:-1]))).sum(dim=1)
        # softmax = torch.div(torch.exp(torch.index_select(x, 1, torch.tensor(self._deepjetindices[:-1]))), sumexp[:, None])
        # print('sumexp')
        # print(sumexp)
        # print('softmax')
        # print(softmax)


        if self._skipone:
            thedeepjets = torch.index_select(x, 1, torch.tensor(self._deepjetindices[:-1], device=x.device))
        else:
            thedeepjets = torch.index_select(x, 1, torch.tensor(self._deepjetindices, device=x.device))

        if self._applylogittransform:
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

        if self._applylogittransform:

            if self._onnxcompatible:

                normalized[normalized < self._eps] = self._eps
                normalized[normalized > 1-self._eps] = 1-self._eps

                normalized = self._logittransformfactor * torch.log(normalized/(1-normalized))

            else:

                normalized = self._logittransformfactor * torch.logit(normalized)  # doesn't work with onnx (opset 13)


        if self._deepjetindices[0] > 0:
            x_ = [torch.index_select(x, 1,
                                     torch.tensor([idx for idx in range(self._deepjetindices[0])], device=x.device)),
                  normalized]
            x = torch.cat(x_, dim=1)
        else:
            x = normalized

        return x


class DeepJetConstraint(nn.Module):
    def __init__(self, deepjetindices, applylogittransform=False, logittransformfactor=1., eps=1e-6):

        super(DeepJetConstraint, self).__init__()

        self._deepjetindices = deepjetindices
        self._applylogittransform = applylogittransform
        self.register_buffer('_logittransformfactor', torch.tensor(logittransformfactor))
        self.register_buffer('_eps', torch.tensor(eps))

    def forward(self, x):

        # TODO: why is there no sigmoid transform here??

        # skip last index because analytical layer increases dimension by 1
        sumexp = torch.exp(torch.index_select(x, 1, torch.tensor(self._deepjetindices[:-1], device=x.device))).sum(
            dim=1)
        softmax = torch.div(
            torch.exp(torch.index_select(x, 1, torch.tensor(self._deepjetindices[:-1], device=x.device))),
            sumexp[:, None])

        analytical = torch.cat((
            torch.index_select(softmax, 1, torch.tensor(0, device=x.device)),  # B
            torch.index_select(softmax, 1, torch.tensor(1, device=x.device)),  # C
            torch.div(torch.index_select(softmax, 1, torch.tensor(1, device=x.device)),
                      torch.index_select(softmax, 1, torch.tensor(1, device=x.device)) + torch.index_select(softmax, 1,
                                                                                                            torch.tensor(
                                                                                                                0,
                                                                                                                device=x.device))),
            # CvB
            torch.div(torch.index_select(softmax, 1, torch.tensor(1, device=x.device)),
                      torch.index_select(softmax, 1, torch.tensor(1, device=x.device)) + torch.index_select(softmax, 1,
                                                                                                            torch.tensor(
                                                                                                                2,
                                                                                                                device=x.device)) + torch.index_select(
                          softmax, 1, torch.tensor(3, device=x.device))),  # CvL
            torch.div(torch.index_select(softmax, 1, torch.tensor(3, device=x.device)),
                      torch.index_select(softmax, 1, torch.tensor(3, device=x.device)) + torch.index_select(softmax, 1,
                                                                                                            torch.tensor(
                                                                                                                2,
                                                                                                                device=x.device))),
        # QG
        ), 1)

        # analytical = torch.full(size=(x.size(dim=0), 5), fill_value=0.2)

        if self._applylogittransform:

            analyticalt = torch.t(analytical)
            analytical_ = torch.empty_like(analyticalt)
            for idim, dim in enumerate(analyticalt):

                result = torch.empty_like(dim)
                for iel, element in enumerate(dim):
                    if element < self._eps:
                        result[iel] = self._logittransformfactor * torch.log(self._eps / (1 - self._eps))
                    elif element > 1 - self._eps:
                        result[iel] = self._logittransformfactor * torch.log((1 - self._eps) / (1 - (1 - self._eps)))
                    else:
                        result[iel] = self._logittransformfactor * torch.log(element / (1 - element))

                analytical_[idim] = result

            analytical = torch.t(analytical_)

            # analytical = torch.logit(analytical, eps=1e-6)
            # analytical = torch.log((analytical+1e-6)/(1-analytical+1e-6))

        # append not supported by onnx with opset<11
        # x_ = []
        #
        # if self._deepjetindices[0] > 0:
        #     x_.append(torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])])))
        #
        # x_.append(analytical)
        #
        # if x.size(dim=1) > self._deepjetindices[-1]:
        #     x_.append(torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[-1], x.size(dim=1))])))
        #
        # x = torch.cat(x_, dim=1)

        # if self._deepjetindices[0] > 0:
        #     if x.size(dim=1) > self._deepjetindices[-1]:
        #         x_ = [torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])])),
        #               analytical,
        #               torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[-1], x.size(dim=1))]))]
        #     else:
        #         x_ = [torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[0])])),
        #               analytical]
        # else:
        #     if x.size(dim=1) > self._deepjetindices[-1]:
        #         x_ = [analytical,
        #               torch.index_select(x, 1, torch.tensor([idx for idx in range(self._deepjetindices[-1], x.size(dim=1))]))]
        #     else:
        #         x_ = [analytical]
        #
        # x = torch.cat(x_, dim=1)

        if self._deepjetindices[0] > 0:
            x_ = [torch.index_select(x, 1,
                                     torch.tensor([idx for idx in range(self._deepjetindices[0])], device=x.device)),
                  analytical]
            x = torch.cat(x_, dim=1)
        else:
            x = analytical

        # xt = torch.t(x)
        # x_ = torch.empty((xt.size(dim=0)+1, xt.size(dim=1)))
        # ix_ = 0
        # for idim in range(self._deepjetindices[0]):
        #     x_[ix_] = torch.index_select(x, 1, torch.tensor(idim))
        #     ix_ += 1
        #
        # for idim in
        #
        # for idim, dim in enumerate(xt):
        #
        #     result = torch.empty_like(dim)
        #     for iel, element in enumerate(dim):
        #         if element < eps:
        #             result[iel] = torch.log(eps / (1 - eps))
        #         elif element > 1 - eps:
        #             result[iel] = torch.log((1 - eps) / (1 - (1 - eps)))
        #         else:
        #             result[iel] = torch.log(element / (1 - element))
        #
        #     x_[idim] = result
        #
        # x = torch.t(x_)

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

            nn.init.normal_(self._linears[-1].weight, mean=0.0, std=0.001)
            nn.init.normal_(self._linears[-1].bias, mean=0.0, std=0.001)
            # nn.init.normal_(self._linears[-1].weight, mean=0.0, std=0.)
            # nn.init.normal_(self._linears[-1].bias, mean=0.0, std=0.)

            self._leakyrelus.append(nn.LeakyReLU())
            if dropout: self._dropouts.append(nn.Dropout(dropout))


        # self._linear = nn.Linear(in_features=in_features, out_features=out_features)
        # nn.init.normal_(self._linear.weight, mean=0.0, std=0.001)
        # nn.init.normal_(self._linear.bias, mean=0.0, std=0.001)
        # # nn.init.zeros_(self._linear.weight)
        # # nn.init.zeros_(self._linear.bias)
        #
        # self._leakyrelu = nn.LeakyReLU()


    def forward(self, x):

        # identity = torch.index_select(x, 1, torch.tensor(self._skipindices, device=x.device))
        # print('')
        # print('identity')
        # print(identity)

        if self._in_features < self._out_features:

            # print('old')
            # print(nn.functional.pad(identity, (0, self._out_features - self._in_features)))
            # print('new')
            # print(torch.cat((identity, torch.zeros(identity.size(dim=0), self._out_features - self._in_features)), dim=1))

            if self._isfirst:
                # TODO: only this would select the wrong elements if skipindices doesn't start at 0, is this also a problem in dense sequential?
                identity = torch.index_select(x, 1, torch.tensor(self._skipindices, device=x.device))
            else:
                identity = torch.index_select(x, 1, torch.tensor([idx for idx in range(len(self._skipindices))], device=x.device))


            # identity = nn.functional.pad(identity, (0, self._out_features - self._in_features))
            identity = torch.cat((identity,
                                  torch.zeros((identity.size(dim=0), self._out_features - len(self._skipindices)),
                                              device=x.device)), dim=1)

        elif self._out_features < self._in_features:

            # gives onnx error...
            # identity = torch.index_select(identity, 1, torch.tensor([idx for idx in range(self._out_features)]))

            # why so complicated??
            # identityt = torch.t(identity)
            # identity_ = torch.empty((self._out_features, identityt.size(dim=1)), device=x.device)
            # for idim, dim in enumerate(identityt):
            #     if idim >= self._out_features: break
            #     identity_[idim] = dim
            #
            # identity = torch.t(identity_)

            if self._islast and len(self._skipindices) > self._n_vars:
                # if we have skipped also the parameters, we need to drop them in the last layer
                identity = x[:, self._n_params:self._n_params+self._out_features]
            else:
                identity = x[:, :self._out_features]  # TODO: why skip all elements and not just len(skipindices)?

        else:

            identity = x


        if self._noskipping:
            identity = torch.zeros_like(identity)

        # print('self._linear(x)')
        # print(self._linear(x))
        # print('identity')
        # print(identity)
        # print('output')

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


class OneHotEncode(nn.Module):
    def __init__(self, source_idx, target_vals, eps=1.):

        super(OneHotEncode, self).__init__()

        self._source_idx = source_idx
        self._target_vals = target_vals
        self.register_buffer('_eps', torch.tensor(eps))

    def forward(self, x):

        # doesn't work with onnx
        # onehot = torch.cat([
        #     self._eps * torch.eq(torch.index_select(x, 1, torch.tensor(self._source_idx, device=x.device)), val).int() for val in self._target_vals
        # ], 1)

        onehot = torch.zeros(x.size(dim=0), len(self._target_vals), device=x.device)
        for ival, val in enumerate(self._target_vals):
            onehot[:, ival:ival+1] = self._eps * (x[:, self._source_idx:self._source_idx+1] == val).int()

        x = torch.cat([
            x[:, :self._source_idx],
            onehot,
            x[:, self._source_idx+1:]
        ], 1)

        return x


class CastTo16Bit(nn.Module):
    def __init__(self):
        super(CastTo16Bit, self).__init__()

    def forward(self, x):

        original_dtype = x.dtype
        x = x.half()
        x[x == 1.] = 0.9995117
        # x[x == 1.] = 1. - torch.finfo(torch.float16).eps
        # x[x == 0.] = 0. + torch.finfo(torch.float16).eps
        return x.to(original_dtype)  # don't _really_ cast to 16bit otherwise dtype problems within the model

        # return x.half().to(x.dtype)  # don't _really_ cast to 16bit otherwise dtype problems within the model


# see https://pytorch.org/docs/1.9.0/_modules/torch/nn/modules/container.html#Sequential
class DenseSequential(nn.Sequential):

    # _identities: List[torch.Tensor]
    # _in_features: List[int]
    # _out_features: List[int]
    # _islasts: List[int]

    def __init__(self, n_params, n_vars, skipindices, in_features_list, out_features_list):

        super(DenseSequential, self).__init__()

        self._n_params = n_params
        self._n_vars = n_vars
        self._skipindices = skipindices

        # self._identities = []
        self._in_features = in_features_list
        self._out_features = out_features_list
        self._islasts = [0 for _ in self._in_features]
        self._islasts[-1] = 1

        # print('in_features')
        # print(self._in_features)
        # print('out_features')
        # print(self._out_features)
        # print('islasts')
        # print(self._islasts)


    def forward(self, input):

        # clear the identities from the last forward pass
        # self._identities.clear()
        identities: List[torch.Tensor] = []  # see https://pytorch.org/docs/1.9.0/jit_language_reference_v2.html#type-annotation

        iblock = 0
        for module in self:

            if isinstance(module, LinearWithSkipConnection):

                if iblock == 0:

                    # print('')
                    # print(iblock)
                    # print(module)
                    # print('input')
                    # print(input.size())
                        
                    for i in range(len(self._islasts)):

                        if self._in_features[iblock] < self._out_features[i]:

                            identity = torch.index_select(input, 1, torch.tensor(self._skipindices, device=input.device))

                            identity = torch.cat((identity,
                                                  torch.zeros((identity.size(dim=0), self._out_features[i] - len(self._skipindices)),
                                                              device=input.device)), dim=1)

                        elif self._out_features[i] < self._in_features[iblock]:

                            if self._islasts[i] == 1 and len(self._skipindices) > self._n_vars:
                                # if we have skipped also the parameters, we need to drop them in the last layer
                                identity = input[:, self._n_params:self._n_params+self._out_features[i]]
                            else:
                                identity = input[:, :self._out_features[i]]

                        else:

                            identity = input

                        # self._identities.append(identity)
                        identities.append(identity)

            input = module(input)

            if isinstance(module, LinearWithSkipConnection):
                input += identities[iblock]
                iblock += 1

        return input



if __name__ == '__main__':

    pass

    # # test = torch.tensor([[100, -3, 0.5, 0.6]])
    # # test = torch.tensor([[0.0172272, 0.0883789, 0.836914, 0.0899048, 0.75293]])
    # test = torch.tensor([[0.474365, 0.398438, 0.598145, 0.27124]])
    # testmodel = nn.Sequential()
    # testmodel.add_module('DeepJetTransform4to4fromNano', DeepJetTransform4to4fromNano(deepjetindices=[0, 1, 2, 3]))
    # testmodel.add_module('DeepJetTransform4to4toNano', DeepJetTransform4to4toNano(deepjetindices=[0, 1, 2, 3]))
    # # testmodel.add_module('DeepJetTransform5to4', DeepJetTransform5to4(deepjetindices=[0, 1, 2, 3, 4]))
    # # testmodel.add_module('DeepJetTransform4to5', DeepJetTransform4to5(deepjetindices=[0, 1, 2, 3, 4]))
    # # testmodel.add_module('Tanh200Transform', TanhTransform(mask=[1, 0, 0, 0], norm=200))
    # # testmodel.add_module('LogitTransform', LogitTransform(mask=[1, 1, 1, 1]))
    # # testmodel.add_module('LogitTransformBack', LogitTransformBack(mask=[1, 1, 1, 1]))
    # # testmodel.add_module('Tanh200TransformBack', TanhTransformBack(mask=[1, 0, 0, 0], norm=200))
    #
    # print('input')
    # print(test)
    #
    # print('train')
    # testmodel.train()
    # print(testmodel(test))
    # print(testmodel(test).sum(dim=1))
    #
    # print('eval')
    # testmodel.eval()
    # print(testmodel(test))
    # print(testmodel(test).sum(dim=1))

    # test = torch.tensor([[10, 0.0172272, 0.0883789, 0.836914, 0.0899048, 0.75293], [1000, 0.0172272, 0.0883789, 0.836914, 0.0899048, 0.75293]])
    # print('test')
    # print(test)
    # print(LinearWithSkipConnection(in_features=6, out_features=10, skipindices=[idx for idx in range(1, 6)]).forward(test))
    # # print(LinearWithSkipConnection(in_features=10, out_features=5).forward(LinearWithSkipConnection(in_features=6, out_features=10).forward(test)))
    #
    # # test = torch.tensor([[0.0172272, 0.0883789, 0.836914, 0.0899048, 0.75293, 0., 0.]])
    # # print('test')
    # # print(test)
    # # print(LinearWithSkipConnection(in_features=7, out_features=5).forward(test))

