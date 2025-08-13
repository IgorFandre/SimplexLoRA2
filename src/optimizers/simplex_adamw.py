import math
import warnings
from typing import Callable

import torch
import torch.optim as optim

def proj_simplex_softmax(x: torch.Tensor, temp=1.0):
    x_0 = (x - x.max()) / temp
    return torch.exp(x_0) / torch.exp(x_0).sum()

def proj_simplex_weighted_softmax(x: torch.Tensor, temp=1.0):
    x_0 = (x - x.max()) / temp
    return (torch.exp(x_0) * x) / (torch.exp(x_0) * x).sum()

def proj_simplex_euclidean(x: torch.Tensor, temp=None, tau=0.0001, max_iter=1000):
    '''
    Bisection for projection onto the simplex
    FROM: http://www.mblondel.org/publications/mblondel-icpr2014.pdf
    temp is not used!
    '''
    null = torch.Tensor([0]).to(x.device)
    func = lambda k: torch.sum(torch.maximum(x - k, null)) - 1
    lower = torch.min(x) - 1 / len(x)
    upper = torch.max(x)

    for _ in range(max_iter):
        midpoint = (upper + lower) / 2.0
        value = func(midpoint)

        if abs(value) <= tau:
            break

        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint

    ans = torch.maximum(x - midpoint, null)
    return ans / torch.sum(ans)

class SimplexAdamW(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay for Simplex Lora adapter

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        lora_r: int = 8
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        self.lora_r = lora_r
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["name"] != "weight_params":
                ############################ Adam Step #############################
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            else:
                ######################## StoIHT step for w #########################
                if group["max_simplex_steps"] == 0:
                    continue
                group["step"] = group.get("step", 0) + 1
                group["adapters"] = group.get("adapters", list(range(group["num_adapters"])))
                group["ranks"] = group.get("ranks", [self.lora_r] * group["num_adapters"])

                w_vector = []
                w_grad = []
                do_proj = False

                for i, p in enumerate(group["params"]):
                    if p.grad is None or i not in group["adapters"]:
                        continue
                
                    p.add_(p.grad, alpha=-group['lr'])
                    if group["wd"] > 0.0:
                        p.add_(p - 1, alpha=(-group["lr"] * group["wd"]))

                    w_vector.append(p.data.item())

                for i, p in enumerate(group["params"]):
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                    state["step"] += 1
                    if state["step"] < group["max_fat_steps"]:
                        p.add_(p.grad, alpha=-group["lr"])
                        w_grad.append(p.grad.item())
                        w_vector.append(p.data.item())
                        do_proj = True

                if do_proj:
                    group["max_simplex_steps"] -= 1
                    j = 0
                    w_grad = torch.tensor(w_grad)
                    if torch.linalg.norm(w_grad).item() > 1e-10:
                        w_vector = torch.tensor(w_vector)
                        w_vector = group["proj"](w_vector, self.temp) * len(group["adapters"])
                        for i, p in enumerate(group["params"]):
                            if p.grad is None or i not in group["adapters"]:
                                continue
                            p.data = torch.tensor([w_vector[j]], device=p.device)
                            j += 1
                    
                        new_adapters = []
                        new_lora_ranks_all = torch.ceil(w_vector * self.default_lora_rank).int()
                        new_lora_ranks_nonzero = []
                        new_lora_weights_nonzero = []
                        
                        j = 0
                        for i, p in enumerate(group["params"]):
                            if p.grad is None or i not in group["adapters"]:
                                continue
                            if new_lora_ranks_all[j] > 0:
                                new_adapters.append(i)
                                new_lora_ranks_nonzero.append(new_lora_ranks_all[i])
                                new_lora_weights_nonzero.append(w_vector[i])
                            j += 1

                        group["adapters"] = new_adapters
                        self.lora_r = new_lora_ranks_nonzero
                

                        for group in self.param_groups:
                            if group["name"] == "weight_params":
                                continue
                            
                
            ####################################################################
        return loss