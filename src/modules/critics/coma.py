import torch
import torch.nn as nn
import torch.nn.functional as F

class COMACritic(nn.Module):
    
    def __init__(self, scheme, args):
        super().__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def _get_input_shape
    