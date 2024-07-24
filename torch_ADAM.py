import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc, _fused_doc)
from typing import List, Optional

__all__ = ['adam', 'ADAM']


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta_one=0.9, beta_two=0.999, epsilon=1e-9, *, maximize: bool = False):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta_one < 0.0:
            raise ValueError(f"Invalid beta_one: {beta_one}")
        if beta_two < 0.0:
            raise ValueError(f"Invalid beta_two: {beta_two}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")

        defaults = dict(lr=lr, beta_one=beta_one, beta_two=beta_two, epsilon=epsilon, maximize=maximize)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                momentum_buffer_list.append(state.get('momentum_buffer'))

        return has_sparse_grad

    @_use_grad_for_differentiable

