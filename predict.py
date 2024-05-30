import torch
from PIL import Image

from torch import nn, load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

# get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1, 28, 28
# classes 0-9

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6) * (28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# instance of nn, loss, optimizerc
# 'cuda' if avail
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr = 1E-3)
loss_fn = nn.CrossEntropyLoss()

# prediction flow
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    #img = Image.open('img_1.jpg') # 2
    #img = Image.open('img_2.jpg') # 0
    img = Image.open('img_3.jpg') # 9
    
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    print(torch.argmax(clf(img_tensor)))
