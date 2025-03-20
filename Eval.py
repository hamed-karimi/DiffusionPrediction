import torch
from DiffusionModel import DiffusionProcess

def predict_image(diffusion_process: DiffusionProcess,
                  model, context, n_timesteps):
    next_image = diffusion_process.generate_next_image( model, context, n_timesteps)
    return next_image

# def visualize_denoising_example(model, sample, device, timesteps, betas):
#     model.eval()
#     with torch.no_grad():
#         # Ensure the test_image is in the correct format (batch size of 1, 3 channels, height, width)
#         test_image = sample['target'].unsqueeze(0)
#         test_image = test_image.to(device)
#         context = sample['context'].unsqueeze(0)
#         context = context.to(device)
#
#         # Add noise (forward process)
#         time = torch.tensor([timesteps]).to(device)
#
#         noisy_image, _ = forward_diffusion(test_image, time=time, betas=betas)  # Add noise at final timestep
#
#         # Denoise the image (reverse process)
#         predicted_noise = model(noisy_image, context, time)  # The model predicts the noise at timestep t
#
#         # Retrieve alpha_t and beta_t for the current timestep (t)
#         beta_t = betas[0] + (betas[1] - betas[0]) * time / timesteps
#         alpha_t = 1 - beta_t
#         sqrt_alpha_t = torch.sqrt(alpha_t)
#         sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
#
#         # Compute the denoised image
#         denoised_image = (noisy_image - sqrt_one_minus_alpha_t.view(-1, 1, 1, 1) * predicted_noise) / sqrt_alpha_t.view(
#             -1, 1, 1, 1)
#
#         # Convert images to numpy arrays for plotting
#         test_image = test_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         noisy_image = noisy_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         denoised_image = denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         previous_image = context[:, :3, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         current_image = context[:, 3:, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#
#         # Clip values to [0, 1] for visualization
#         test_image = np.clip(test_image, 0, 1)
#         noisy_image = np.clip(noisy_image, 0, 1)
#         denoised_image = np.clip(denoised_image, 0, 1)
#         previous_image = np.clip(previous_image, 0, 1)
#         current_image = np.clip(current_image, 0, 1)
#
#         # Plotting the original, noisy, and denoised image
#         fig, axes = plt.subplots(2, 3, figsize=(12, 4))
#
#         axes[0, 0].imshow(previous_image)
#         axes[0, 0].set_title("Previous Frame")
#         axes[0, 0].axis("off")
#
#         axes[0, 1].imshow(current_image)
#         axes[0, 1].set_title("Current Frame")
#         axes[0, 1].axis("off")
#
#         axes[0, 2].imshow(test_image)
#         axes[0, 2].set_title("Next Frame")
#         axes[0, 2].axis("off")
#
#         axes[1, 0].imshow(noisy_image)
#         axes[1, 0].set_title("Noisy Next Frame")
#         axes[1, 0].axis("off")
#
#         axes[1, 1].imshow(denoised_image)
#         axes[1, 1].set_title("Predicted Next Frame")
#         axes[1, 1].axis("off")
#
#         axes[1, 2].axis("off")
#
#         plt.show()
#
#
# def visualize_predictive_example(model, sample, device, timesteps, betas):
#     model.eval()
#     with torch.no_grad():
#         # Ensure the test_image is in the correct format (batch size of 1, 3 channels, height, width)
#         test_image = sample['target'].unsqueeze(0)
#         test_image = test_image.to(device)
#         context = sample['context'].unsqueeze(0)
#         context = context.to(device)
#
#         # Add noise (forward process)
#         t = torch.tensor([timesteps]).to(device)
#
#         noisy_image, _ = forward_diffusion(torch.zeros_like(test_image), t=t, timesteps=timesteps,
#                                            betas=betas)  # Add noise at final timestep
#
#         time_tiled = get_tiled_time(t, test_image)
#         print(time_tiled.shape, test_image.shape)
#         # Denoise the image (reverse process)
#         predicted_noise = model(noisy_image, context, time_tiled)  # The model predicts the noise at timestep t
#
#         # Retrieve alpha_t and beta_t for the current timestep (t)
#         beta_t = betas[0] + (betas[1] - betas[0]) * t / timesteps
#         alpha_t = 1 - beta_t
#         sqrt_alpha_t = torch.sqrt(alpha_t)
#         sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
#
#         # Compute the denoised image
#         denoised_image = (noisy_image - sqrt_one_minus_alpha_t.view(-1, 1, 1, 1) * predicted_noise) / sqrt_alpha_t.view(
#             -1, 1, 1, 1)
#
#         # Convert images to numpy arrays for plotting
#         test_image = test_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         noisy_image = noisy_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         denoised_image = denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         previous_image = context[:, :3, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#         current_image = context[:, 3:, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#
#         # Clip values to [0, 1] for visualization
#         test_image = np.clip(test_image, 0, 1)
#         noisy_image = np.clip(noisy_image, 0, 1)
#         denoised_image = np.clip(denoised_image, 0, 1)
#         previous_image = np.clip(previous_image, 0, 1)
#         current_image = np.clip(current_image, 0, 1)
#
#         correlation_target = np.corrcoef(denoised_image.reshape(-1), test_image.reshape(-1))[0, 1]
#         correlation_current = np.corrcoef(denoised_image.reshape(-1), current_image.reshape(-1))[0, 1]
#         correlation_previous = np.corrcoef(denoised_image.reshape(-1), previous_image.reshape(-1))[0, 1]
#
#         # Plotting the original, noisy, and denoised image
#         fig, axes = plt.subplots(2, 3, figsize=(12, 4))
#
#         axes[0, 0].imshow(previous_image)
#         axes[0, 0].set_title("Previous Frame")
#         axes[0, 0].axis("off")
#
#         axes[0, 1].imshow(current_image)
#         axes[0, 1].set_title("Current Frame")
#         axes[0, 1].axis("off")
#
#         axes[0, 2].imshow(test_image)
#         axes[0, 2].set_title("Next Frame")
#         axes[0, 2].axis("off")
#
#         axes[1, 0].imshow(noisy_image)
#         axes[1, 0].set_title("Noisy Next Frame")
#         axes[1, 0].axis("off")
#
#         axes[1, 1].imshow(denoised_image)
#         axes[1, 1].set_title("Predicted Next Frame")
#         axes[1, 1].axis("off")
#
#         axes[1, 2].axis("off")
#
#         plt.show()
#
#         print('Correlation with next frame: ', correlation_target)
#         print('Correlation with current frame: ', correlation_current)
#         print('Correlation with previous frame: ', correlation_previous)
#
#
# def estimate_correlations(model, dataset, device, timesteps, betas):
#     model.eval()
#     with torch.no_grad():
#         all_corr_target = []
#         all_corr_current = []
#         all_corr_previous = []
#         # for idx in range(dataset.__len__()):
#         for idx in range(200):
#             sample = dataset.__getitem__(idx)
#             test_image = sample['target'].unsqueeze(0)
#             test_image = test_image.to(device)
#             context = sample['context'].unsqueeze(0)
#             context = context.to(device)
#
#             t = torch.tensor([timesteps]).to(device)
#             time_tiled = get_tiled_time(t, test_image)
#             noisy_image, _ = forward_diffusion(torch.zeros_like(test_image), t=t, timesteps=timesteps,
#                                                betas=torch.tensor(betas).to(device))  # Add noise at final timestep
#             predicted_noise = model(noisy_image, context, time_tiled)
#             # Retrieve alpha_t and beta_t for the current timestep (t)
#             beta_t = betas[0] + (betas[1] - betas[0]) * t / timesteps
#             alpha_t = 1 - beta_t
#             sqrt_alpha_t = torch.sqrt(torch.tensor(alpha_t))
#             sqrt_one_minus_alpha_t = torch.sqrt(1 - torch.tensor(alpha_t))
#
#             # Compute the denoised image
#             denoised_image = (noisy_image - sqrt_one_minus_alpha_t.view(-1, 1, 1,
#                                                                         1) * predicted_noise) / sqrt_alpha_t.view(-1, 1,
#                                                                                                                   1, 1)
#
#             # Convert images to numpy arrays for plotting
#             test_image = test_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#             noisy_image = noisy_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#             denoised_image = denoised_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#             previous_image = context[:, :3, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#             current_image = context[:, 3:, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
#             correlation_target = np.corrcoef(denoised_image.reshape(-1), test_image.reshape(-1))[0, 1]
#             correlation_current = np.corrcoef(denoised_image.reshape(-1), current_image.reshape(-1))[0, 1]
#             correlation_previous = np.corrcoef(denoised_image.reshape(-1), previous_image.reshape(-1))[0, 1]
#             all_corr_target.append(correlation_target)
#             all_corr_current.append(correlation_current)
#             all_corr_previous.append(correlation_previous)
#     return all_corr_target, all_corr_current, all_corr_previous