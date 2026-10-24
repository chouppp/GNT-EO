"""Process-shareable Adam optimizer for CPU A3C workers."""

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(parameter.data)
                state["exp_avg_sq"] = torch.zeros_like(parameter.data)

    def share_memory(self):
        for state in self.state.values():
            state["step"].share_memory_()
            state["exp_avg"].share_memory_()
            state["exp_avg_sq"].share_memory_()
