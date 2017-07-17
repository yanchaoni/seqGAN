import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, cuda=False):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.cuda = cuda

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.cuda:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        # input dim                                             # batch_size x seq_len
        emb = self.embeddings(input)                            # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                              # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                       # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(hidden.view(-1, self.hidden_dim))    # batch_size x 1
        out = F.sigmoid(out)
        return out

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)