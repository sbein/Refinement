"""The Modified Differential Multiplier Method (MDMM) for PyTorch.

adapted from https://github.com/crowsonkb/mdmm
see https://www.engraved.blog/how-we-can-make-machine-learning-algorithms-tunable/
"""

import abc
from dataclasses import dataclass
from typing import List

import torch
from torch import nn, optim


@dataclass
class ConstraintReturn:
    """The return type for constraints."""
    value: torch.Tensor
    fn_value: torch.Tensor
    inf: torch.Tensor


class Constraint(nn.Module, metaclass=abc.ABCMeta):
    """The base class for all constraint types."""

    def __init__(self, fn, scale, damping, lmbda_init=0.):
        super().__init__()
        self.fn = fn
        self.register_buffer('scale', torch.as_tensor(scale))
        self.register_buffer('damping', torch.as_tensor(damping))
        self.lmbda = nn.Parameter(torch.tensor(lmbda_init))

    def extra_repr(self):
        return f'scale={self.scale:g}, damping={self.damping:g}'

    @abc.abstractmethod
    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        ...

    def forward(self, inp, out, target, allow_adaptive_epsilon=False):
        fn_value = self.fn(inp, out, target)
        inf = self.infeasibility(fn_value, allow_adaptive_epsilon)
        l_term = self.lmbda * inf
        damp_term = self.damping * inf**2 / 2

        # print('scale', self.scale)
        # print('value', self.value)
        # print('fn_value', fn_value)
        # print('inf', inf)
        # print('l_term', l_term)
        # print('damp_term', damp_term)

        return ConstraintReturn(self.scale * (l_term + damp_term), fn_value, inf)


class EqConstraint(Constraint):
    """Represents an equality constraint."""

    def __init__(self, fn, value, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('value', torch.as_tensor(value))

    def extra_repr(self):
        return f'value={self.value:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        return self.value - fn_value


class EqConstraintAdaptiveEpsilon(Constraint):
    """Represents an equality constraint."""

    def __init__(self, fn, value, scale=1., damping=1., lmbda_init=0., look_back=100):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('value', torch.as_tensor(value))

        self.register_buffer('look_back', torch.as_tensor(look_back))
        self.register_buffer('last_fn_values', torch.empty((0)))

    def extra_repr(self):
        return f'value={self.value:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):

        if allow_adaptive_epsilon:
            with torch.no_grad():

                if self.last_fn_values is not None:

                    if not self.last_fn_values.device == fn_value.device:
                        self.last_fn_values = self.last_fn_values.to(fn_value.device)

                    self.last_fn_values = torch.cat((self.last_fn_values, fn_value.unsqueeze(0)))

                    if self.last_fn_values.size()[0] >= 3 * self.look_back:
                        if self.last_fn_values[-self.look_back:].mean() \
                                > self.last_fn_values[-2*self.look_back:-self.look_back].mean() \
                                > self.last_fn_values[-3*self.look_back:-2*self.look_back].mean():
                            if self.last_fn_values[-3*self.look_back:-2*self.look_back].mean() < self.value:
                                self.value = self.last_fn_values[-3*self.look_back:-2*self.look_back].mean()
                                self.last_fn_values = None
                                print('epsilon adapted to ' + str(self.value.item()))

                    # if self.last_fn_values.size()[0] >= self.look_back and self.last_fn_values[-self.look_back:].mean() < self.value:
                    #     self.value = self.last_fn_values[-self.look_back:].mean()
                    #     self.last_fn_values = torch.empty((0))
                    #     print('epsilon adapted to ' + str(self.value.item()))

        return self.value - fn_value


class MaxConstraint(Constraint):
    """Represents a maximum inequality constraint which uses a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('max', torch.as_tensor(max))
        self.slack = nn.Parameter(torch.as_tensor(float('nan')))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        if self.slack.isnan():
            with torch.no_grad():
                self.slack.copy_((self.max - fn_value).relu().pow(1/2))
        return self.max - fn_value - self.slack**2


class MaxConstraintHard(Constraint):
    """Represents a maximum inequality constraint without a slack variable."""

    def __init__(self, fn, max, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'max={self.max:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        return fn_value.clamp(max=self.max) - fn_value


class MinConstraint(Constraint):
    """Represents a minimum inequality constraint which uses a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('min', torch.as_tensor(min))
        self.slack = nn.Parameter(torch.as_tensor(float('nan')))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        if self.slack.isnan():
            with torch.no_grad():
                self.slack.copy_((fn_value - self.min).relu().pow(1/2))
        return fn_value - self.min - self.slack**2


class MinConstraintHard(Constraint):
    """Represents a minimum inequality constraint without a slack variable."""

    def __init__(self, fn, min, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('min', torch.as_tensor(min))

    def extra_repr(self):
        return f'min={self.min:g}, scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        return fn_value.clamp(min=self.min) - fn_value


class BoundConstraintHard(Constraint):
    """Represents a bound constraint."""

    def __init__(self, fn, min, max, scale=1., damping=1., lmbda_init=0.):
        super().__init__(fn, scale, damping, lmbda_init)
        self.register_buffer('min', torch.as_tensor(min))
        self.register_buffer('max', torch.as_tensor(max))

    def extra_repr(self):
        return f'min={self.min:g}, max={self.max:g}, ' \
               f'scale={self.scale:g}, damping={self.damping:g}'

    def infeasibility(self, fn_value, allow_adaptive_epsilon=False):
        return fn_value.clamp(self.min, self.max) - fn_value


@dataclass
class MDMMReturn:
    """The return type for MDMM."""
    value: torch.Tensor
    fn_values: List[torch.Tensor]
    infs: List[torch.Tensor]


class MDMM(nn.ModuleList):
    """The main MDMM class, which combines multiple constraints."""

    def make_optimizer(self, params, *, optimizer=optim.Adamax, lr=2e-3, lr_lambda_factor=1.):
        lambdas = [c.lmbda for c in self]
        slacks = [c.slack for c in self if hasattr(c, 'slack')]
        return optimizer([{'params': params, 'lr': lr},
                          {'params': lambdas, 'lr': -lr_lambda_factor*lr},
                          {'params': slacks, 'lr': lr}])

    def forward(self, loss, inp, out, target, switch_off_constraints=False, allow_adaptive_epsilon=False):
        value = loss.clone()
        fn_values, infs = [], []
        for c in self:
            c_return = c(inp, out, target, allow_adaptive_epsilon)
            if not switch_off_constraints: value += c_return.value
            fn_values.append(c_return.fn_value)
            infs.append(c_return.inf)
        return MDMMReturn(value, fn_values, infs)


if __name__ == '__main__':

    pass
