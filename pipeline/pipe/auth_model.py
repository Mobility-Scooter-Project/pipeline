# authentication model
# inputs: (1, 27, 128)
# output: ?


import os
import torch
from ..utils.auth.modelComponents import init_siamese

class AuthModel:
    def __init__(self):
        self.model  = init_siamese(deg = 3)
        state_dict = torch.load(os.path.join("assets", "model_state_dict.pth"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def process(self, inputs):
        with torch.no_grad():
            res = self.model(inputs.unsqueeze(dim = 0))
            res = res.detach().squeeze(dim  = 0).numpy().tolist()
            return res

    def __del__(self):
        pass
