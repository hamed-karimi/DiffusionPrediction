import torch
import torch.nn as nn
import torch.optim as optim
from DiffusionModel import DiffusionProcess
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F


def train_model(device,
                diffusion_process: DiffusionProcess,
                dataloader, model, optimizer, n_epochs, n_timesteps):
    # Training loop
    track = []
    scaler = GradScaler(enabled=True)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, sample in enumerate(dataloader):

            x_0 = sample['target']
            # x_0 = x_0.to(device)
            context = sample['context']
            # context = context.to(device)

            # Sample a random timestep t
            time = torch.randint(0, n_timesteps, (x_0.shape[0],))

            # Add noise (forward process)
            # noisy_x_t, true_noise = forward_diffusion(x_0, time, betas)
            noisy_x_t, true_noise = diffusion_process.forward_diffusion(x_0, time)
            if torch.isnan(noisy_x_t).any():
                print("NaNs detected in xt")
                break

            # Calculate loss (denoising the noisy image)
            optimizer.zero_grad()
            # loss = loss_fn(x_0, x_t, context, model, t, timesteps, betas, true_noise)

            with autocast(enabled=True):
                noise_pred = model(noisy_x_t.to(device),
                                   context.to(device),
                                   time.to(device))
                loss = F.l1_loss(noise_pred, true_noise)

            scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # loss.backward()

            if torch.isnan(loss).any():
                print("NaNs detected in loss")
                break

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer.step()

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loss_value = loss.detach().cpu().numpy()
            track.append(loss_value)

            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{n_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f"Epoch [{epoch + 1}/{n_epochs}] completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")
    return model, optimizer, track


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)