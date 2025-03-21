import torch
import numpy as np
import math


def cosine_beta_schedule(n_timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)

def linear_beta_schedule(n_timesteps, beta_start=0.0, beta_end=0.0001):
    betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)
    return torch.clip(betas, beta_start, beta_end)

class DiffusionProcess:

    def __init__(self, n_timesteps, beta_start, beta_end):
        self.timesteps = np.arange(n_timesteps)
        self.betas = linear_beta_schedule(n_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_0, time):
        alpha_cumprod_t = self.alphas_cumprod[time]
        noise = torch.randn_like(x_0)  # Create random noise

        # True noise used for generating the noisy image
        true_noise = noise

        # Noisy image at timestep t
        noisy_x_t = torch.sqrt(alpha_cumprod_t).view(-1, 1, 1, 1) * x_0 + torch.sqrt(1-alpha_cumprod_t).view(-1, 1, 1, 1) * noise
        # x_t = torch.sqrt(1 - beta_t).view(-1, 1, 1, 1) * x_0 + torch.sqrt(beta_t).view(-1, 1, 1, 1) * noise

        return noisy_x_t, true_noise  # Return noisy image and true noise

    def one_step_denoise(self, model, noisy_x_t, context, timesteps):
        model.eval()
        with torch.no_grad():
            predicted_noise = model(noisy_x_t, context, timesteps.to(context.device))  # The model predicts the noise at timestep t -> 0

        alpha_t = self.alphas[timesteps].to(noisy_x_t.device)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1).to(noisy_x_t.device)
        beta_t = self.betas[timesteps].view(-1, 1, 1, 1).to(noisy_x_t.device)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[timesteps]).view(-1, 1, 1, 1).to(noisy_x_t.device)
        denoised_image = (noisy_x_t - (beta_t/sqrt_one_minus_alpha_t) * predicted_noise) / sqrt_alpha_t

        return denoised_image

    def generate_next_image(self, model, context, n_timesteps):
        generator = torch.manual_seed(0)
        print('context device', context.device)
        assert context.dim() == 4, "Context must be a 4D tensor for image generation"
        # x_t = torch.randn((1, 3, context.shape[2], context.shape[3]),
        #                   generator=generator)
        init_x_t = torch.ones((1, 3, context.shape[2], context.shape[3]))
        x_t, _ = self.forward_diffusion(init_x_t, n_timesteps)
        x_t = x_t.to(context.device)
        x_t_list = [x_t]
        for t in reversed(range(n_timesteps)):
            if t > 0:
                z = torch.randn((1, 3, context.shape[2], context.shape[3]),
                                generator=generator)
                z = z.to(context.device)
            else:
                z = 0
            beta_t = self.betas[t].to(context.device)
            alpha_t = self.alphas[t].to(context.device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(context.device)
            with torch.no_grad():
                pred_noise = model(x_t, context, torch.Tensor([t]).to(context.device))
            x_t_1 = (x_t - pred_noise * beta_t / torch.sqrt(1-alpha_cumprod_t)) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * z

            x_t = x_t_1.clone()
            x_t_list.append(x_t)

        return x_t, x_t_list