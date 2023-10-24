import torch
import torch.nn as nn

# Resuidual Block
class ResidualBlock(nn.Module):
    def __init__(self, input1, features):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input1, features,3, padding = 'same'), 
            nn.BatchNorm2d(features), 
            nn.ReLU(), 
            nn.Conv2d(features, input1, 3, padding = 'same'),
            nn.BatchNorm2d(input1))
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        F = self.conv(x)
        H = F + x
        return self.relu(H)
    
    
# Full CNN 
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,64, 7, stride=2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            nn.Conv2d(64,128, 3, stride=2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #28
            
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        
            #layer 3
            nn.Conv2d(128,256, 3, stride=2, padding = 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #7
            
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            
            #layer 4
            nn.Conv2d(256,512, 3, stride=2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            ResidualBlock(512, 152),
            ResidualBlock(512, 152),
            ResidualBlock(512, 152),
            ResidualBlock(512, 152),
         
            nn.AvgPool2d(2,1),
        
            nn.Dropout(p=dropout),
   
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
