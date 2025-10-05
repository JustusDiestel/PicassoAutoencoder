from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import Autoencoder as AE
import torch.nn as nn
import torch


def get_dataloaders():
    transform = transforms.ToTensor()
    data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(data, batch_size=64, shuffle=True)
    test_loader = DataLoader(data, batch_size=64, shuffle=False)
    return train_loader, test_loader

def show_batch(train_loader):
    images, labels = next(iter(train_loader))
    print(images.shape)
    print(labels.shape)
    plt.figure(figsize=(12, 2))
    for i in range(12):
        plt.subplot(1, 12, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
    plt.show()






def main():
    train_loader, test_loader = get_dataloaders()
    show_batch(train_loader)

    model = AE.Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, _ in train_loader:
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

            if epoch % 10 == 9:
                plt.figure(figsize=(4, 2))
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(images[0].detach().squeeze(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Rekonstruktion")
                plt.imshow(outputs[0].detach().squeeze(), cmap='gray')
                plt.axis('off')

                plt.show()
if __name__ == '__main__':
    main()