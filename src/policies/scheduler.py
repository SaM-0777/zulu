from typing import Any
import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas=False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Any,
        sample: torch.Tensor,
        to_final: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_sample(
        self,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        sample_stabilized: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stabilized) / sigma
        return model_output

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        sigma = self.sigmas[timestep_id].to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(
        self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        target = noise - sample
        return target

    def training_weight(self, timestep: torch.Tensor):
        timestep_id = torch.argmin(
            (
                self.timesteps.unsqueeze(1)
                - timestep.unsqueeze(0).to(self.timesteps.device)
            ).abs(),
            dim=0,
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
