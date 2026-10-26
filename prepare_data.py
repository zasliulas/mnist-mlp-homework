
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def prepare_data(show_image=True):
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataset, val_dataset = random_split(
        train_full,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    print("Train örnek sayısı:", len(train_dataset))
    print("Validation örnek sayısı:", len(val_dataset))
    print("Test örnek sayısı:", len(test_dataset))

    images, labels = next(iter(train_loader))
    print("Batch görüntü boyutu:", images.shape)
    print("Batch etiket boyutu:", labels.shape)
    print("İlk 10 etiket:", labels[:10].tolist())

    if show_image:
        image = images[0].squeeze() * 0.3081 + 0.1307
        plt.imshow(image, cmap="gray")
        plt.title(f"Gerçek etiket: {labels[0].item()}")
        plt.axis("off")
        plt.show()

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    prepare_data()
