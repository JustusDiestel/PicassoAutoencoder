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

def main():
    train_loader, test_loader = get_dataloaders()


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
            last_images = images
            last_outputs = outputs
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.subplot(2, 10, i + 11)
            plt.imshow(outputs[i].detach().squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()
if __name__ == '__main__':
    main()