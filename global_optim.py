import torch
from settings import logger, formatter

class GlobalOptimizer:
    def __init__(self, model, lr, beta1=0.9, beta2=0.99, eps=1e-8):
        self.outdated_params = {name: p.clone().detach() for name, p in model.state_dict().items()}
        self.lr = lr
        logger.info(f'Global learning rate: {self.lr}')
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # Initialize momentum and precondition for each parameter
        self.exp_avg = {name: torch.zeros_like(p) for name, p in self.outdated_params.items()}
        self.exp_avg_sq = {name: torch.zeros_like(p) for name, p in self.outdated_params.items()}
        self.step = 0
    
    def update_outdated_params(self, new_params):
        # update the outdated params
        self.outdated_params = new_params
    
    def update_momentum_precondition(self, model_diffs):
        self.step += 1
        # logger.info(f'number of keys model_diffs:{len(model_diffs.keys())} exp_avg:{len(self.exp_avg.keys())} exp_avg_sq: {len(self.exp_avg_sq.keys())}')
        for name, diff in model_diffs.items():
            if name in self.exp_avg and name in self.exp_avg_sq:
                self.exp_avg[name] = self.beta1 * self.exp_avg[name] + (1 - self.beta1) * diff
                self.exp_avg_sq[name] = self.beta2 * self.exp_avg_sq[name] + (1 - self.beta2) * (diff ** 2)

    def update_param(self, model_diffs):
        self.update_momentum_precondition(model_diffs)
        
        updated_params = {}
        for name, p in self.outdated_params.items():
            corrected_exp_avg = self.exp_avg[name] / (1 - self.beta1 ** self.step)
            corrected_exp_avg_sq = self.exp_avg_sq[name] / (1 - self.beta2 ** self.step)

            # Update parameters
            denom = corrected_exp_avg_sq.sqrt().add_(self.eps)
            step_size = self.lr * (corrected_exp_avg / denom)
            updated_params[name] = p + step_size
            
        return updated_params


    # def zero_grad(self):
    #     """Zero all gradients in the network manually."""
    #     for p in self.params:
    #         if p.grad is not None:
    #             p.grad.detach_()
    #             p.grad.zero_()
