import torch.optim as optim
import sys
#from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, adam_sania, muon, diag_hvp

def get_optimizer(args, model):
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params       = model.parameters(),
            lr           = args.lr,
            betas        = (args.beta1, args.beta2),
            eps          = args.eps,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "soap":
        optimizer = soap.SOAP(
            params                 = model.parameters(),
            lr                     = args.lr,
            betas                  = (args.beta1, args.beta2),
            shampoo_beta           = args.shampoo_beta,
            eps                    = args.eps,
            weight_decay           = args.weight_decay,
            precondition_frequency = args.update_freq,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            params       = model.parameters(),
            lr           = args.lr,
            momentum     = args.momentum,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "adam-sania":
        optimizer = adam_sania.AdamSania(
            params       = model.parameters(),
            lr           = args.lr,
            betas        = (args.beta1, args.beta2),
            eps          = args.eps,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "muon":
        optimizer = muon.Muon(
            muon_params  = list(model.parameters()),
            lr           = args.lr,
            adamw_betas  = (args.beta1, args.beta2),
            adamw_eps    = args.eps,
            adamw_wd     = args.weight_decay,
            momentum     = args.momentum,
            ns_steps     = args.ns_steps,
        )
    elif args.optimizer == "diag-hvp":
        optimizer = diag_hvp.DiagonalPreconditionedOptimizer(
            params = model.parameters(),
            lr = args.lr,
            eps = args.eps,
            update_freq = args.update_freq
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    return optimizer
