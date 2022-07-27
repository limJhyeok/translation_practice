import torch
import copy

def run(model, train_batch, val_batch, loss_fn, optimizer, num_epochs, device, print_every = 100, clip = 1):
  '''
  wrapper of train and evaluation
  '''
  min_loss = float('inf')
  best_model = None
  for epoch in range(num_epochs):
    total_train_loss = train(model, train_batch, optimizer, loss_fn, device, clip)
    total_val_loss = evaluation(model, val_batch, loss_fn, device)
    if (epoch+1) % print_every == 0 or epoch == 0:
      print(f'Epoch| {epoch+1}/{num_epochs}')
      print(f'train loss: {total_train_loss/len(train_batch[0])}')
      print(f'val loss: {total_val_loss/len(val_batch[0])}')

    if min_loss > total_val_loss:
      min_loss = total_val_loss
      best_model = copy.deepcopy(model)

  return best_model

def train(model, batch, optimizer, loss_fn, device, clip = 1 ):
  '''
  train one epoch
  inputs:
    - model
    - batch(Tuple(src Tensor, trg Tensor)): Tensor.size() = (seq_length, batch_size)
    - optimizer
    - loss_fn
    - device: GPU or CPU
    - clip(float): protect gardient exploding by limiting max norm of gradient
  outputs:
    -  running_loss(float): total loss in train one epoch
  '''
  model.train()
  model = model.to(device)
  loss_fn = loss_fn.to(device)
  running_loss = 0.0
  for i, (src, trg) in enumerate(zip(batch[0], batch[1])):
    src, trg = src.to(device), trg.to(device) 
    logits = model(src, trg) # logits: [seq_length, batch_size,  n_tokens]
    n_tokens = logits.size(-1)
    logits = logits[1:].reshape(-1, n_tokens) # remove <sos> token and flatten (seq_length * batch_size - 1, n_tokens)
    trg = trg[1:].reshape(-1) # (seq_length * batch_size - 1)

    optimizer.zero_grad()
    loss = loss_fn(logits, trg)
    running_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
  return running_loss

def evaluation(model, batch, loss_fn, device):
  '''
  evaluation one epoch
  inputs:
    - model
    - batch(Tuple): (src Tensor, trg Tensor) (*Tensor.size() = (seq_length, batch_size))
    - loss_fn
    - device: GPU or CPU
  outputs:
    -  running_loss(float): total loss in evaluation one epoch
  '''
  model.eval()
  model = model.to(device)
  loss_fn = loss_fn.to(device)
  running_loss = 0.0
  with torch.no_grad():
    for i, (src, trg) in enumerate(zip(batch[0], batch[1])):
      src, trg = src.to(device), trg.to(device)
      logits = model(src, trg) # logits: [seq_length, batch_size,  n_tokens]
      n_tokens = logits.size(-1)
      logits = logits[1:].reshape(-1, n_tokens) # remove <sos> token and flatten (seq_length * batch_size - 1, n_tokens)
      trg = trg[1:].reshape(-1) # (seq_length * batch_size - 1)
      loss = loss_fn(logits, trg)
      running_loss += loss.item()

  return running_loss



