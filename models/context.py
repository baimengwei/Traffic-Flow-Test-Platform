import torch
import torch.nn as nn
import torch.nn.functional as F

class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """

    def __init__(self, hidden_sizes=[50], output_dim=None,
                 input_dim=None, only_concat_context=0,
                 hidden_activation=F.relu, history_length=1,
                 action_dim=None, obsr_dim=None, device='cpu'):
        """
        Args:
            hidden_sizes: [H]
            output_dim: H
            input_dim: A + R + S
            only_concat_context: 3
            hidden_activation: e.g. F.relu
            history_length: 15, 25 or 30, based on args and cmd setting
            action_dim: A
            obsr_dim: S
            device:
        """
        super(Context, self).__init__()
        self.only_concat_context = only_concat_context
        self.hid_act = hidden_activation
        self.fcs = []
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        # count the fact that there is a skip connection
        self.output_dim_final = output_dim
        self.output_dim_last_layer = output_dim // 2
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim

        # build LSTM or multi-layers FF
        if only_concat_context == 3:
            # use LSTM or GRU
            # A + R + S -> H
            self.recurrent = nn.GRU(self.input_dim,
                                    self.hidden_sizes[0],
                                    bidirectional=False,
                                    batch_first=True,
                                    num_layers=1)

    def init_recurrent(self, bsize=None):
        """
            init hidden states
            Batch size can't be none
        """
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        """
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        """
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]

        if self.only_concat_context == 3:
            # first prepare data for LSTM
            # previous_action is B* (history_len * D)
            bsize, dim = previous_action.shape
            # view(bsize, self.hist_length, -1)
            pacts = previous_action.view(bsize, -1, self.action_dim)
            # reward dim is 1, view(bsize, self.hist_length, 1)
            prews = previous_reward.view(bsize, -1, 1)
            # view(bsize, self.hist_length, -1)
            pxs = pre_x.view(bsize, -1, self.obsr_dim)
            # input to LSTM is [action, reward]
            pre_act_rew = torch.cat([pacts, prews, pxs], dim=-1)

            # init lstm/gru. H init.
            hidden = self.init_recurrent(bsize=bsize)

            # lstm/gru
            # hidden is (1, B, hidden_size)
            _, hidden = self.recurrent(pre_act_rew, hidden)
            out = hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)

            return out
        else:
            raise NotImplementedError
