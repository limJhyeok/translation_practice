import torch
import torch.nn as nn
import random

#device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')

class Seq2Seq(nn.Module):
  def __init__(self,
               encoder,
               decoder,
               device):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, src, trg, teacher_forcing_ratio = 0.75):
    '''
    inputs:
      - src(Tensor[src_seq_length, batch_size])
      - trg(Tensor[trg_seq_length, batch_size])
      - teacher_forcing_ratio(float): input of decoder will be ground truths token or prediction following by ratio
    '''

    enc_outs, hidden = self.encoder(src)
    seq_len = trg.size(0)
    batch_size = trg.size(1)
    trg_n_tokens = self.decoder.trg_n_tokens

    logits = torch.zeros((seq_len, batch_size, trg_n_tokens)).to(self.device)
    input = trg[0, :] # input(Tensor[batch_size])
    for i in range(1, seq_len):
      output, hidden = self.decoder(input, hidden, enc_outs)
      logits[i] = output
      top1 = output.argmax(1)
      if self.training: # We should not use teacher forcing when eval
        input = trg[i] if random.random() < teacher_forcing_ratio else top1
      else:
        input = top1
    return logits