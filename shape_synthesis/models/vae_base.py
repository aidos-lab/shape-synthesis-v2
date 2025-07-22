import torch
from dect.nn import EctConfig
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_dim=28, hidden_dim=400, latent_dim=200):
        super().__init__()
        input_dim = in_dim * in_dim
        output_dim = in_dim * in_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ###################################################
            # (B, 128, W) -> (B, 64, W/2)
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=(2, 14)),
            nn.ReLU(),
            ###################################################
            # (B, 64, W/2) -> (B, 32, W/4)
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normalized_shape=(4, 7)),
            nn.ReLU(),
            ###################################################
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=(4, 7)),
            nn.ReLU(),
            ###################################################
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=(512, 32)),
            nn.ReLU(),
            ###################################################
            # (B, 64, W/4) -> (B, 32, W/8)
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            # Output: (B,1024,16)
        )

        self.decoder = nn.Sequential(
            ###################################################
            nn.ConvTranspose2d(
                8, 4, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.LayerNorm(normalized_shape=(512, 32)),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose1d(
                512, 512, kernel_size=7, stride=1, padding=3, output_padding=0
            ),
            nn.LayerNorm(normalized_shape=(512, 32)),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose1d(
                512, 512, kernel_size=7, stride=1, padding=3, output_padding=0
            ),
            nn.LayerNorm(normalized_shape=(512, 32)),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose1d(
                512, 256, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.LayerNorm(normalized_shape=(256, 64)),
            nn.ReLU(),
            ###################################################
            nn.ConvTranspose1d(
                256, 128, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(1024 * 16, latent_dim)
        self.fc_var = nn.Linear(1024 * 16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 1024 * 16)

    def forward_encoder(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance

        return mean, log_var

    def forward_decoder(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon  # reparameterization trick
        return z

    def sample(self, n):
        sample = torch.randn(n, self.latent_dim, device="cuda")
        return self.forward_decoder(sample).view(-1, 1, 28, 28)

    def forward(self, x):
        mean, log_var = self.forward_encoder(x.flatten(start_dim=1))
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.forward_decoder(z).view(-1, 1, 28, 28)

        return x_hat, mean, log_var


if __name__ == "__main__":
    model = VAE()
    x = torch.rand(size=(2, 28, 28))
    print(model(x))
    x_hat, _, _ = model(x)
    print(x_hat.shape)
