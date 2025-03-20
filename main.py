from Train import train_model, init_weights
import torch
import DiffusionModel
import os
from Utils import create_video_dict
from PaperClipDataset import PaperclipDataset
from torch.utils.data import DataLoader
from UNet import AdvancedUNet
from torch import optim

if __name__ == '__main__':
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    learning_rate = 1e-4
    epochs = 30
    betas_endpoints = (1e-04, 0.02)  # or (1e04, 0.02)
    n_timesteps = 150

    folder_path1 = './minidataset'
    # folder_path2 = '/mmfs1/data/projects/sccn/shared/transfers/objectsDiffusion/minidataset2'
    Testing_data_folder = './minidataset_testing'
    video_dict = create_video_dict(folder_path1)
    dataset = PaperclipDataset(video_dict)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Adjust based on your CPU
        pin_memory=True  # Useful when using CUDA
    )

    diffusion_process = DiffusionModel.DiffusionProcess(n_timesteps=n_timesteps,
                                                        beta_start=betas_endpoints[0],
                                                        beta_end=betas_endpoints[1])
    model = AdvancedUNet().to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer, track = train_model(device=device,
                                          dataloader=dataloader,
                                          diffusion_process=diffusion_process,
                                          optimizer=optimizer,
                                          model=model,
                                          n_epochs=epochs,
                                          n_timesteps=n_timesteps)
    torch.save(model.state_dict(), 'predictive_diffusion_model.pth')
