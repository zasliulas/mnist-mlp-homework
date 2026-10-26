
import torch
import torch.nn as nn


class MNIST_MLP_Experiment2(nn.Module):
    def __init__(self):
        super(MNIST_MLP_Experiment2, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = MNIST_MLP_Experiment2()
    sample = torch.randn(1, 1, 28, 28)
    output = model(sample)

    print(model)
    print("Çıktı boyutu:", output.shape)
    print(
        "Parametre sayısı:",
        sum(p.numel() for p in model.parameters())
    )
