import torch
import torch.nn as nn

class ECG_CNN(nn.Module):
    def __init__(self, n_samples=187, num_classes=5):
        super().__init__()

        ##########
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )

        ##########
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )

        ##########
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )

        ##########
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),
        )

        ##########
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=256 * (n_samples // 16),
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features=128,
                out_features=num_classes,
            )
        )

        ####################
        self.x_to_conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), 
        )

        self.conv1_to_conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), 
        )

    def forward(self, x):
        
        # x: (B, 1, 187)
        # (batch_size, channel_dim, sequence_length)

        # out1: (B, 32, 187//2)
        out1 = self.conv1(x)

        # out2: (B, 64, 187//4)
        out2 = self.conv2(out1)

        # skip1: x -> conv2 output 
        # (B, 1, 187) -> (B, 64, 187//4)
        skip1 = self.x_to_conv2(x)

        # out3: (B, 128, 187//8)
        out3 = self.conv3(out2 + skip1)

        # skip2: out1 -> conv3 output 
        # (B, 32, 187//2) -> (B, 128, 187//8)
        skip2 = self.conv1_to_conv3(out1)

        # out4: (B, 256, 187//16)
        out4 = self.conv4(out3 + skip2)

        # out: (B, 256, 187//32)
        out = self.fc(out4)

        return out

if __name__ == "__main__":
    model = ECG_CNN()

    # Test with dummy input
    x = torch.randn(2, 1, 187)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total paramenters: {sum(p.numel() for p in model.parameters())}")