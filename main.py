import Eval
from Train import train_model, init_weights
import torch
import DiffusionModel
import os
from Utils import create_video_dict
from PaperClipDataset import PaperclipDataset
from torch.utils.data import DataLoader
from UNet import AdvancedUNet
from torch import optim
import pickle

if __name__ == '__main__':
    mode = 'Train'
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_workers = 4
    learning_rate = 1e-5
    epochs = 100
    betas_endpoints = (1e-04, 0.02)  # or (1e04, 0.02)
    n_timesteps = 250

    # folder_path1 = './minidataset'
    # Testing_data_folder = './minidataset_testing'

    folder_path1 = '/mmfs1/data/projects/sccn/shared/transfers/objectsDiffusion/minidataset'
    folder_path2 = '/mmfs1/data/projects/sccn/shared/transfers/objectsDiffusion/minidataset2'
    Testing_data_folder = '/mmfs1/data/projects/sccn/shared/transfers/objectsDiffusion/minidataset_testing'

    video_dict = create_video_dict(folder_path1)
    dataset = PaperclipDataset(video_dict)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Adjust based on your CPU
        pin_memory=True  # Useful when using CUDA
    )

    diffusion_process = DiffusionModel.DiffusionProcess(n_timesteps=n_timesteps,
                                                        beta_start=betas_endpoints[0],
                                                        beta_end=betas_endpoints[1])
    model = AdvancedUNet().to(device)
    if mode == 'Train':
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

        # Save track using pickle
        with open('training_track.pkl', 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(track, f)

    if mode == 'Test':
        infer_time = 100
        model.load_state_dict(torch.load('predictive_diffusion_model.pth', map_location=device))
        test_image = dataset.__getitem__(7)
        context = test_image['context'].unsqueeze(0)
        target = test_image['target'].unsqueeze(0)
        # noisy = diffusion_process.forward_diffusion(target.to(device), torch.tensor([infer_time - 1]))[0]
        # next_pred = diffusion_process.one_step_denoise(model, noisy.to(device), context.to(device),
        #                                                torch.tensor([infer_time - 1]))
        # denoised_next_pred_img = next_pred[0].permute([1, 2, 0]).cpu().numpy()

        # generated_image = Eval.predict_image(diffusion_process, model, context, infer_time)
        corr_prev, corr_curr, corr_next = Eval.prediction_correlation(diffusion_process, 1., device, model, dataset, infer_time)