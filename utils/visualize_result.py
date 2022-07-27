import random
import torch

def print_val(best_model, input_lang, output_lang, val_batch, device):
  num_print = 10
  pick = random.randint(0, len(val_batch[0]) - 1)
  with torch.no_grad():
    best_model.eval()
    best_model = best_model.to(device)
    src, trg = val_batch[0][pick], val_batch[1][pick]
    src, trg = src.to(device), trg.to(device)
    logits = best_model(src, trg)
    preds = logits.argmax(-1)

    # [seq_length, batch_size] -> [batch_size, seq_length] (for simply treating sentences)
    preds = preds.T
    src = src.T
    trg = trg.T
    
    for i in range(num_print):
      input_sentence = [input_lang.index2word[src.item()] for src in src[i]]
      answer = [output_lang.index2word[trg.item()] for trg in trg[i]]
      pred = [output_lang.index2word[pred.item()] for pred in preds[i]]
      print('source sentence: ', ' '.join(input_sentence[1:]))
      print('answer translation: ', ' '.join(answer[1:]))
      print('pred translation : ', ' '.join(pred[1:]))
      print(' ')