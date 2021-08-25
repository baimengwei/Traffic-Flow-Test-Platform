from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import get_action_info

##################################
# Actor and Critic Newtork for TD3
##################################


class Actor(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """

    def __init__(self, action_space, hidden_sizes=[400, 300],
                 input_dim=None, hidden_activation=F.relu,
                 max_action=None, enable_context=False,
                 hiddens_dim_conext=[50], input_dim_context=None,
                 output_conext=None, only_concat_context=0,
                 history_length=1, obsr_dim=None, device='cpu'
                 ):
        """
        Args:
            action_space: env.action_space
            hidden_sizes: e.g. [300, 300]
            input_dim: actor_idim, e.g. [S + H] or [S]
            hidden_activation:
            max_action: float(env.action_space.high[0])
            enable_context: True/ False, use True.
            hiddens_dim_conext: H
            input_dim_context: A + R + S or None
            output_conext: H or 0
            only_concat_context: should be 3
            history_length: 15, 25 or 30, based on args and cmd setting.
            obsr_dim: env.observation_space.shape[0]
            device: cpu.
        """
        super(Actor, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)

        self.actor = nn.Sequential(
            nn.Linear(input_dim[0], self.hsize_1),
            nn.ReLU(),
            nn.Linear(self.hsize_1, self.hsize_2),
            nn.ReLU()
        )
        self.out = nn.Linear(self.hsize_2, action_dim)
        self.max_action = max_action
        self.enable_context = enable_context
        self.output_conext = output_conext

        # context network
        self.context = None
        if self.enable_context:
            self.context = Context(hidden_sizes=hiddens_dim_conext,
                                   input_dim=input_dim_context,
                                   output_dim=output_conext,
                                   only_concat_context=only_concat_context,
                                   hidden_activation=hidden_activation,
                                   history_length=history_length,
                                   action_dim=action_dim,
                                   obsr_dim=obsr_dim,
                                   device=device
                                   )

    def forward(self, x, pre_act_rew=None, state=None, ret_context=False):
        """
            input (x  : B * D where B is batch size and D is input_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        """
        combined = None
        if self.enable_context:
            # A + R + S -> H
            combined = self.context(pre_act_rew)
            x = torch.cat([x, combined], dim=-1)

        x = self.actor(x)
        x = self.max_action * torch.tanh(self.out(x))

        if ret_context:
            return x, combined

        else:
            return x

    def get_conext_feats(self, pre_act_rew):
        """
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        """
        combined = self.context(pre_act_rew)
        return combined


class Critic(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """

    def __init__(self, action_space, hidden_sizes=[400, 300], input_dim=None,
                 hidden_activation=F.relu, enable_context=False,
                 dim_others=0, hiddens_dim_conext=[50], input_dim_context=None,
                 output_conext=None, only_concat_context=0, history_length=1,
                 obsr_dim=None, device='cpu'):
        """
        References:
            Actor Class.
        Args:
            action_space:
            hidden_sizes:
            input_dim:
            hidden_activation:
            enable_context:
            dim_others:
            hiddens_dim_conext:
            input_dim_context:
            output_conext:
            only_concat_context:
            history_length:
            obsr_dim:
            device:
        """
        super(Critic, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)

        # handling extra dim
        self.enable_context = enable_context

        if self.enable_context:
            self.extra_dim = dim_others  # right now, we add reward + previous action

        else:
            self.extra_dim = 0

        # It uses two different Q networks
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(input_dim[0] + action_dim + self.extra_dim,
                      self.hsize_1),
            nn.ReLU(),
            nn.Linear(self.hsize_1, self.hsize_2),
            nn.ReLU(),
            nn.Linear(self.hsize_2, 1),
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(input_dim[0] + action_dim + self.extra_dim,
                      self.hsize_1),
            nn.ReLU(),
            nn.Linear(self.hsize_1, self.hsize_2),
            nn.ReLU(),
            nn.Linear(self.hsize_2, 1),
        )

        if self.enable_context:
            self.context = Context(hidden_sizes=hiddens_dim_conext,
                                   input_dim=input_dim_context,
                                   output_dim=output_conext,
                                   only_concat_context=only_concat_context,
                                   history_length=history_length,
                                   action_dim=action_dim,
                                   obsr_dim=obsr_dim,
                                   device=device
                                   )

    def forward(self, x, u, pre_act_rew=None, ret_context=False):
        """
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        """
        xu = torch.cat([x, u], 1)
        combined = None

        if self.enable_context:
            combined = self.context(pre_act_rew)
            xu = torch.cat([xu, combined], dim=-1)
        # Q1
        x1 = self.q1(xu)
        # Q2
        x2 = self.q2(xu)

        if ret_context:
            return x1, x2, combined

        else:
            return x1, x2

    def Q1(self, x, u, pre_act_rew=None, ret_context=False):
        """
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        """
        xu = torch.cat([x, u], 1)
        combined = None

        if self.enable_context:
            combined = self.context(pre_act_rew)
            xu = torch.cat([xu, combined], dim=-1)

        # Q1
        x1 = self.q1(xu)

        if ret_context:
            return x1, combined

        else:
            return x1

    def get_conext_feats(self, pre_act_rew):
        '''
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        '''
        combined = self.context(pre_act_rew)

        return combined


