
import torch
import torch.nn as nn


class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


if __name__ == "__main__":
    model = MNIST_MLP()

    sample = torch.randn(1, 1, 28, 28)
    output = model(sample)

    print(model)
    print("Girdi boyutu:", sample.shape)
    print("Çıktı boyutu:", output.shape)
    print("Toplam parametre sayısı:",
          sum(p.numel() for p in model.parameters()))
