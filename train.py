import torch
from models import Model
from data import get_dataloader, get_val_dataloader
from settings import num_epochs, gpu_ids, path_to_model
from utils import Log
from tqdm import tqdm
import os


device = torch.device(gpu_ids[0])
model = Model().to(device)
if len(gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
dataset = get_dataloader()
train_len = len(dataset)
val_dataset = get_val_dataloader()
val_len = len(val_dataset)

log = Log()
for epoch_n in range(1, num_epochs+1):

    train_loss = 0
    batches = enumerate(dataset)
    for _ in tqdm(range(train_len)):
        model.train()
        i, batch = next(batches)
        loss = model(batch['noisy'].to(device), batch['clean'].to(device),
                     batch['src_pad_mask'].to(device), batch['tgt_pad_mask'].to(device))
        loss = loss.mean()
        print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()
    train_loss /= train_len

    val_loss = 0
    batches = enumerate(val_dataset)
    for _ in tqdm(range(val_len)):
        model.eval()
        i, batch = next(batches)
        with torch.no_grad():
            loss = model(batch['noisy'].to(device), batch['clean'].to(device),
                         batch['src_pad_mask'].to(device), batch['tgt_pad_mask'].to(device))
            loss = loss.mean()
            val_loss += loss.cpu().item()
    val_loss /= val_len

    torch.save(model.state_dict(), os.path.join(path_to_model, 'model'+str(epoch_n)))
    log.write_log(epoch_n, train_loss, val_loss)
    scheduler.step()

log.save_plots()
