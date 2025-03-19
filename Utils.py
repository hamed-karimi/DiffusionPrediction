import os
import numpy as np

def create_frame_pairs(folder_path):
    frame_pairs = {}
    frame_list = sorted(os.listdir(folder_path))  # List of all frame filenames
    for filename in frame_list:
        if filename.endswith('.jpg'):
            frame_number = int(filename[6:-4])  # Extract frame number
            current_frame = os.path.join(folder_path, filename)

            # Previous frame
            if frame_number > 0:
                prev_frame = os.path.join(folder_path, f"frame_{frame_number-1:04d}.jpg")
                frame_pairs[current_frame] = frame_pairs.get(current_frame, {})
                frame_pairs[current_frame]["previous_frame"] = prev_frame

            # Next frame
            next_frame = os.path.join(folder_path, f"frame_{frame_number+1:04d}.jpg")
            if os.path.exists(next_frame):
                frame_pairs[current_frame] = frame_pairs.get(current_frame, {})
                frame_pairs[current_frame]["next_frame"] = next_frame

    return frame_pairs

def create_video_dict(parent_folder):
    video_dict = {}
    for video_folder in os.listdir(parent_folder):
        video_folder_path = os.path.join(parent_folder, video_folder)
        # Ensure it's a directory
        if os.path.isdir(video_folder_path):
            # Create frame pairs for each video folder
            frame_pairs = create_frame_pairs(video_folder_path)
            # Update video_dict with pairs from this video folder
            video_dict.update(frame_pairs)
    return video_dict


def shift_frame_up(frame, shift_amount):
    frame_np = frame.cpu().numpy()
    batch_size, channels, height, width = frame_np.shape
    if shift_amount >= height:
        shift_amount = height - 1
    shifted_frame = np.roll(frame_np, -shift_amount, axis=2) # Shift vertically up
    shifted_frame[:, :, -shift_amount:, :] = 1  # Set the bottom `shift_amount` rows to white (1)

    shifted_frame = np.clip(shifted_frame, 0.0, 1.0)
    shifted_frame_tensor = torch.tensor(shifted_frame).float().to(frame.device)
    return shifted_frame_tensor

# Sample a prediction
# def visualize_predictions(model, dataloader, timesteps=25, save_dir = 'DMVisualizations'):
#     os.makedirs(save_dir, exist_ok=True)
#     device = next(model.parameters()).device
#     checkpoint = torch.load('/mmfs1/data/berkelaa/Senior Thesis Research/DOP2.0/2020.ckpt')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     beta = checkpoint['beta']
#     alpha = checkpoint['alpha']
#     alpha_cumprod = checkpoint['alpha_cumprod']
#     #beta, alpha, alpha_cumprod = compute_alpha_beta(timesteps, device=device)
#     model.eval()

#     with torch.no_grad():
#           for i, (current_frame, next_frame) in enumerate(dataloader):
#             if i>=5:
#                 break
#             x_0 = current_frame.cuda()
#             next_frame = next_frame.cuda()

#             timestep = torch.randint(0, len(alpha_cumprod) - 1, (x_0.size(0),), dtype=torch.long, device=device)

#             x_t, true_noise = forward_diffusion_process(x_0, timestep, alpha, alpha_cumprod, beta, device=device)
#             predicted_image = reverse_diffusion_process(model, x_t, timestep, alpha, alpha_cumprod, beta, device)

#             shifted_frame = shift_frame_up(x_0[:, 3:, :, :], 86).to(device)

#             current_frame_np = current_frame.cpu().numpy()
#             next_frame_np = next_frame.cpu().numpy()
#             predicted_frame_np = predicted_image.cpu().numpy()
#             shifted_frame_np = shifted_frame.cpu().numpy()

#             predicted_frame_np = np.clip(predicted_frame_np, 0, 1)
#             #normalization for better visualization
#             current_frame_np = (current_frame_np - current_frame_np.min()) / (current_frame_np.max() - current_frame_np.min())

#             predicted_frame_np = (predicted_frame_np - predicted_frame_np.min()) / (predicted_frame_np.max() - predicted_frame_np.min())

#             next_frame_np = (next_frame_np - next_frame_np.min()) / (next_frame_np.max() - next_frame_np.min())


#             shifted_frame_np = (shifted_frame_np - shifted_frame_np.min()) / (shifted_frame_np.max() - shifted_frame_np.min() )

#             # Plotting
#             fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#             axs[0, 0].imshow(current_frame_np[0, 3:, :, :].transpose(1, 2, 0))  # Show kast 3 channels (RGB))
#             axs[0, 0].set_title(f'Current Frame {i+1}')
#             axs[0, 0].axis('off')

#             axs[0, 1].imshow(predicted_frame_np[0].transpose(1, 2, 0))  # Show the predicted frame
#             axs[0, 1].set_title(f'Predicted Frame {i+1}')
#             axs[0, 1].axis('off')

#             axs[1, 0].imshow(next_frame_np[0].transpose(1, 2, 0))  # Show the actual next frame
#             axs[1, 0].set_title(f'Actual Next Frame {i+1}')
#             axs[1, 0].axis('off')

#             axs[1, 1].imshow(shifted_frame_np[0].transpose(1, 2, 0))  # Shifted Frame
#             axs[1, 1].set_title(f'Shifted Frame {i + 1}')
#             axs[1, 1].axis('off')

#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f'DOP2.0w25_{i+1}.png'))
#             plt.close()
