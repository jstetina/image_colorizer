import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

class ColorNet(nn.Module):
    DEFAULT_CHECKPOINT_PATH = "checkpoint/colornet.pt"

    def __init__(self, checkpoint_path:str=DEFAULT_CHECKPOINT_PATH):
        super(ColorNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # to scale the output to [0, 1]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        if os.path.exists(checkpoint_path):
           self._load_model(checkpoint_path) 

    def _load_model(self, path):
        print("Loading ColorNet model...", end="")
        self.load_state_dict(torch.load(path, map_location=self.device))
        print("done.")

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, _ in train_loader:
                gray_images = transforms.Grayscale(num_output_channels=1)(inputs).to(self.device)
                gray_images = gray_images.repeat(1,3,1,1)
                color_images = inputs.to(self.device)

                optimizer.zero_grad()

                outputs = model(gray_images)
                loss = criterion(outputs, color_images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * gray_images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        torch.save(model.state_dict(), self.DEFAULT_CHECKPOINT_PATH)


    def colorize(self, input_path:str, output_path):
        input_image = Image.open(input_path).convert("RGB")
        input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_image_tnsr = self(input_image)
            output_image_tnsr = output_image_tnsr.squeeze(0).cpu()
            output_image_tnsr = transforms.ToPILImage()(output_image_tnsr)

            output_image_tnsr.save(output_path)

    def visualize_results(model, test_loader, num_images=5):
        model.eval()
        with torch.no_grad():
            data_iter = iter(test_loader)
            images, _ = data_iter.next()
            
            # Get grayscale and colorized images
            gray_images = images[:num_images]
            colorized_images = model(gray_images)
            
            # Plotting the results
            for i in range(num_images):
                plt.subplot(3, num_images, i+1)
                plt.imshow(gray_images[i].permute(1, 2, 0).squeeze(), cmap="gray")
                plt.axis('off')

                plt.subplot(3, num_images, num_images+i+1)
                plt.imshow(colorized_images[i].permute(1, 2, 0))
                plt.axis('off')

                plt.subplot(3, num_images, 2*num_images+i+1)
                plt.imshow(gray_images[i].permute(1, 2, 0).repeat(3, 1, 1).permute(1, 2, 0))
                plt.axis('off')

            plt.show()


