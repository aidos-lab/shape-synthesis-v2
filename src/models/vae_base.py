import torch
from dect.nn import EctConfig
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_dim=120, hidden_dim=400, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ###################################################
            # (B, 128, W) -> (B, 64, W/2)
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ###################################################
            # (B, 64, W/2) -> (B, 32, W/4)
            nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            # (B, 64, W/4) -> (B, 32, W/8)
            nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3),
            # Output: (B,1024,16)
        )

        self.decoder = nn.Sequential(
            ###################################################
            nn.ConvTranspose2d(
                512, 256, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose2d(
                256, 256, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose2d(
                256, 256, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose2d(
                256, 256, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose2d(
                256, 128, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose2d(
                128, 1, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(512 * 16, latent_dim)
        self.fc_var = nn.Linear(512 * 16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 16)

    def encode(self, input_tensor):
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result_new = self.decoder(result.view(-1, 512, 4, 4))
        return result_new

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_tensor):
        mu, log_var = self.encode(input_tensor.unsqueeze(1))
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def sample(self, n: int, device: str = "cuda"):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(n, self.latent_dim, device=device)

        samples = self.decode(z)
        return samples
