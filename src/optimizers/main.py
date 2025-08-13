import torch.optim as optim
import sys

# from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, adam_sania, muon, diag_hvp, weight_adamw, simplex_adamw


def get_optimizer(args, model):
    if hasattr(args, "ft_strategy") and \
        args.ft_strategy == "SimplexLoRA" and \
            args.optimizer not in ["simplex_adamw"]:
                raise ValueError("Optimizer must be 'simplex_adamw' when using 'SimplexLoRA' strategy.")
    if hasattr(args, "ft_strategy") and \
        args.optimizer in ["simplex_adamw"] and \
            args.ft_strategy != "SimplexLoRA":
                raise ValueError("The 'simplex_adamw' optimizer must be used with the 'SimplexLoRA' strategy.")

    if hasattr(args, "ft_strategy") and \
        args.ft_strategy == "WeightLoRA" and \
            args.optimizer not in ["weight_adamw"]:
                raise ValueError("Optimizer must be 'weight_adamw' when using 'WeightLoRA' strategy.")
    if hasattr(args, "ft_strategy") and \
        args.optimizer in ["weight_adamw"] and \
            args.ft_strategy != "WeightLoRA":
                raise ValueError("The 'weight_adamw' optimizer must be used with the 'WeightLoRA' strategy.")

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "soap":
        optimizer = soap.SOAP(
            params=model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_freq,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam-sania":
        optimizer = adam_sania.AdamSania(
            params=model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "muon":
        optimizer = muon.Muon(
            muon_params=list(p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
        )
    elif args.optimizer == "diag-hvp":
        optimizer = diag_hvp.DiagonalPreconditionedOptimizer(
            params=model.parameters(),
            lr=args.lr,
            eps=args.eps,
            update_freq=args.update_freq,
        )
    elif args.optimizer in ["weight_adamw", "simplex_adamw"]:
        weight_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_weight" in name:
                weight_params.append(param)
            elif "lora_A" in name or "lora_B" in name:
                other_params.append(param)
        if args.optimizer == "weight_adamw":
            optimizer = weight_adamw.WeightAdamW(
                [
                    {"params": other_params, "name": "other_params"},
                    {
                        "params": weight_params,
                        "k": args.k,
                        "proj": weight_adamw.proj_0,
                        "lr": args.lr_w,
                        "max_fat_steps": args.max_fat_steps,
                        "name": "weight_params",
                    },
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "simplex_adamw":
            optimizer = simplex_adamw.SimplexAdamW(
                [
                    {"params": other_params, "name": "other_params"},
                    {
                        "params": weight_params,
                        "proj": simplex_adamw.proj_simplex_euclidean,
                        "lr": args.learning_rate_w,
                        "wd": args.weight_decay_w,
                        "num_adapters": len(other_params),
                        "simplex_step": args.simplex_step,
                        "max_simplex_steps": args.max_simplex_steps,
                        "name": "weight_params",
                    },
                ],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                lora_r=args.lora_r
            )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    return optimizer


# optim_params = []
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         layer_name = name.replace("lora_A.", "").replace("lora_B.", "") # можно поиграться
#         layer_exist = False
#         for param_group in optim_params:
#             if param_group["name"] == layer_name:
#                 param_group["params"].append(param)
#                 layer_exist = True
#         if not layer_exist:
#             optim_params.append({"params": [param], "name": layer_name})
