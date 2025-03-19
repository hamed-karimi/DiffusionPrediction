import torch


def forward_diffusion(x_0, time, betas):
    """Add noise to the image at each timestep, and return the noisy image along with the true noise."""
    # beta_t = betas[0] + (betas[1] - betas[0]) * t / timesteps  # Linear schedule
    beta_t = betas[time]
    noise = torch.randn_like(x_0)  # Create random noise

    # True noise used for generating the noisy image
    true_noise = noise

    # Noisy image at timestep t
    x_t = torch.sqrt(1 - beta_t).view(-1, 1, 1, 1) * x_0 + torch.sqrt(beta_t).view(-1, 1, 1, 1) * noise

    return x_t, true_noise  # Return noisy image and true noise