import math

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc, _fused_doc)
from typing import List, Optional

__all__ = ['adam']


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta_one=0.9, beta_two=0.999, epsilon=1e-9, *, maximize: bool = False,
                 differentiable: bool = False,):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta_one < 0.0:
            raise ValueError(f"Invalid beta_one: {beta_one}")
        if beta_two < 0.0:
            raise ValueError(f"Invalid beta_two: {beta_two}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")

        self.t = 0

        defaults = dict(lr=lr, beta_one=beta_one, beta_two=beta_two, epsilon=epsilon, maximize=maximize,
                        differentiable=differentiable)

        super().__init__(params, defaults)

    def __setstate__(self, state):

        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, d_p_list, m_0_buffer_list, v_0_buffer_list):

        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]

                m_0_buffer_list.append(state.get('m_0_buffer'))
                v_0_buffer_list.append(state.get('v_0_buffer'))

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            m_0_buffer_list = []
            v_0_buffer_list = []
            self.t += 1
            t = self.t
            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list,  m_0_buffer_list, v_0_buffer_list)

            adam(params_with_grad, d_p_list, m_0_buffer_list, v_0_buffer_list, t, beta_one=group['beta_one'],
                 beta_two=group['beta_two'], epsilon=group['epsilon'], lr=group['lr'], maximize=group['maximize'],
                 has_sparse_grad=has_sparse_grad, grad_scale=getattr(self, "grad_scale", None),
                 found_inf=getattr(self, "found_inf", None))

            for p, m_0_buffer, v_0_buffer in zip(params_with_grad, m_0_buffer_list, v_0_buffer_list):
                state = self.state[p]
                state['m_0_buffer'] = m_0_buffer
                state['v_0_buffer'] = v_0_buffer

        return loss


def adam(params: List[Tensor], d_p_list: List[Tensor], m_0_buffer_list: List[Optional[Tensor]],
         v_0_buffer_list: List[Optional[Tensor]], t: int, has_sparse_grad: bool = None,
         grad_scale: Optional[Tensor] = None, found_inf: Optional[Tensor] = None, *, beta_one: float,
         beta_two: float, lr: float, epsilon: float, maximize: bool):

    func = _single_tensor_adam

    func(params, d_p_list, m_0_buffer_list, v_0_buffer_list, t, beta_one=beta_one, beta_two=beta_two,
         epsilon=epsilon, lr=lr, has_sparse_grad=has_sparse_grad, maximize=maximize, grad_scale=grad_scale,
         found_inf=found_inf)


def _single_tensor_adam(params: List[Tensor], d_p_list: List[Tensor], m_0_buffer_list: List[Optional[Tensor]],
                        v_0_buffer_list: List[Optional[Tensor]], t: int, grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor], *, beta_one: float, beta_two: float,
                        lr: float, epsilon: float, maximize: bool, has_sparse_grad: bool):

    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        m_0_buff = m_0_buffer_list[i]

        if m_0_buff is None:
            m_0_buff = torch.zeros_like(d_p)
            m_0_buffer_list[i] = m_0_buff
        else:
            m_0_buff.mul_(beta_one).add_(d_p, alpha=1-beta_one)

        v_0_buff = v_0_buffer_list[i]
        if v_0_buff is None:
            v_0_buff = torch.zeros_like(d_p)
            v_0_buffer_list[i] = v_0_buff
        else:
            v_0_buff.mul_(beta_two).add_(torch.pow(d_p, 2), alpha=1-beta_two)

        lr_t = lr * ((math.sqrt(1-math.pow(beta_two, t))) / (1 - math.pow(beta_one, t)))
        param.add_(torch.div(m_0_buff, torch.add(torch.sqrt(v_0_buff),epsilon)), alpha=-lr_t)


Adam.__doc__ = (r"""Implements Adam optimization method based on 'ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION'.
                Author of code: Jordi Granja Bayot.
                """ + rf"""
                Args:
                    params (iterable): iterable of parameters to optimize or dicts defining
                        parameter groups
                    lr (float, optional): learning rate (default: 1e-3)
                    beta_one (float, optional): beta_one_factor (default: 0.9)
                    beta_two (float, optional): beta_two_factor (default: 0.999)
                    epsilon (float, optional): epsilon_factor (default: 1e-9)
                    {_maximize_doc}
                    {_differentiable_doc}
                """



































    
)




