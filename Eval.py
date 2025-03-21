import torch
from DiffusionModel import DiffusionProcess
import numpy as np

def predict_image(diffusion_process: DiffusionProcess,
                  model, context, n_timesteps):
    next_image, x_t_list = diffusion_process.generate_next_image(model, context, n_timesteps)
    return next_image, x_t_list

def prediction_correlation(diffusion_process: DiffusionProcess, device,model, dataset, n_timesteps):
    n_sample = 50
    sample_inds = np.random.choice(len(dataset), n_sample, replace=False)
    all_corr_prev = []
    all_corr_curr = []
    all_corr_next = []
    for i in sample_inds:
        test_image = dataset.__getitem__(i)
        context = test_image['context'].unsqueeze(0).to(device)
        target = test_image['target'].unsqueeze(0).to(device)
        generated_image, _ = predict_image(diffusion_process, model, context, n_timesteps)
        corr_prev = np.corrcoef(context[:, :3, :, :].squeeze().cpu().numpy().flatten(),
                                generated_image.squeeze().cpu().numpy().flatten())[0, 1]
        all_corr_prev.append(corr_prev)

        corr_curr = np.corrcoef(context[:, 3:, :, :].squeeze().cpu().numpy().flatten(),
                                generated_image.squeeze().cpu().numpy().flatten())[0, 1]
        all_corr_curr.append(corr_curr)

        corr_next = np.corrcoef(target.squeeze().cpu().numpy().flatten(),
                                generated_image.squeeze().cpu().numpy().flatten())[0, 1]
        all_corr_next.append(corr_next)

    return all_corr_prev, all_corr_curr, all_corr_next
