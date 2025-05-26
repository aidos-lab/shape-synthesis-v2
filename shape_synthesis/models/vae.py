import torch
from dect.nn import EctConfig
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_dim=28, hidden_dim=400, latent_dim=200):
        super().__init__()
        input_dim = in_dim * in_dim
        output_dim = in_dim * in_dim
        self.latent_dim = latent_dim

        # Encoder
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Decoder
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

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
