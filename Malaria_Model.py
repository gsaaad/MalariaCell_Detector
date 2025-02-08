from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class Malaria_Model(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=2):
        super().__init__()
        channels, height, width = input_shape

        # Convolutional blocks
        self.conv_block1 = ConvBlock(channels, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)

        # Compute flattened size after 3 downsampling steps (each reduces dimension by 2)
        conv_reduction = 2 ** 3
        flat_height = height // conv_reduction
        flat_width = width // conv_reduction
        flat_size = 128 * flat_height * flat_width

        # Fully connected feature extraction
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True)
        )

        # Separate branches for classification and regression
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for the model using common best practices
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        features = self.fc(x)
        return self.classifier(features), self.regressor(features)

