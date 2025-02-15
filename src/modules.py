import torch
from torch import nn


class BiAffine(nn.Module):
    """
    Biaffine attention layer (Dozat and Manning, 2017):

    The biaffine attention layer takes as input two tensors of shape: [batch_size, sequence_length, biaffine_size]

    Return scores matrices of shape :
        - [batch_size, sequence_length, sequence_length] for arc scores
        - [batch_size, number_of_labels, sequence_length, sequence_length] for label scores
    """

    def __init__(self, head_size, dep_size, scores_per_arc=1, bias=False, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(BiAffine, self).__init__()
        self.scores_per_arc = scores_per_arc
        self.head_size = head_size
        self.dep_size = dep_size
        self.bias = bias
        if self.bias:
            self.head_size += 1
            self.dep_size += 1

        self.U = nn.Parameter(torch.empty(self.scores_per_arc, head_size, dep_size, device=device))

        self.reset_parameters()

        self.device = device

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)

    def forward(self, H_head, H_dep):
        """
        :param H_head: Head tensor of shape [batch_size, sequence_length, head_size]
        :param H_dep: Dependent head tensor of shape [batch_size, sequence_length, dependent_size]
        :return: a score matrix S of shape:
            - if self.scores_per_arc == 1 (for arc score): [batch_size, sequence_length, sequence_length]
            - if self.scores_per_arc > 1 (for label score): [batch_size, nb_labels, sequence_length, sequence_length]
                with nb_labels == scores_per_arc.
        """
        bias = None
        if self.bias:
            b = H_head.shape[0]
            L = H_head.shape[1]
            bias = torch.ones(b, L, 1, device=self.device)

        # Add dimension in input tensors for nb_labels dimension
        # (cf. otherwise broadcast will be prepended 1 dim of size 1)
        H_head = H_head.unsqueeze(1)
        H_dep = H_dep.unsqueeze(1)

        S = H_head @ self.U @ H_dep.transpose(-1, -2)

        return S.squeeze(1) + bias
