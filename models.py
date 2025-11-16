import torch
import torch.nn as nn
import torchvision.models as models


class GraspingCNN(nn.Module):
    """Custom CNN model for grasp prediction"""

    def __init__(self, input_channels=3, output_dim=3):
        """
        Args:
            input_channels: Number of input image channels (RGB=3)
            output_dim: Output dimension (x, y, angle) = 3
        """
        super(GraspingCNN, self).__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        # Input image 224x224, after 4 maxpool (2x2), feature map size is 14x14
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward propagation"""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class GraspingResNet(nn.Module):
    """ResNet-based grasp prediction model"""

    def __init__(self, output_dim=3, pretrained=True):
        """
        Args:
            output_dim: Output dimension (x, y, angle) = 3
            pretrained: Whether to use pretrained weights
        """
        super(GraspingResNet, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Get ResNet feature dimension
        num_features = self.resnet.fc.in_features

        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        """Forward propagation"""
        return self.resnet(x)


class GraspingVGG(nn.Module):
    """VGG-based grasp prediction model (optional)"""

    def __init__(self, output_dim=3, pretrained=True):
        """
        Args:
            output_dim: Output dimension (x, y, angle) = 3
            pretrained: Whether to use pretrained weights
        """
        super(GraspingVGG, self).__init__()

        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)

        # Get VGG feature dimension
        num_features = self.vgg.classifier[0].in_features

        # Replace classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward propagation"""
        return self.vgg(x)


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test models"""

    print("=" * 70)
    print("               🧪 Testing Models")
    print("=" * 70)

    # Create random input (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)

    # Test CNN
    print("\n1️⃣  Testing GraspingCNN...")
    cnn_model = GraspingCNN()
    cnn_output = cnn_model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {cnn_output.shape}")
    print(f"   Parameters:   {count_parameters(cnn_model):,}")
    print(f"   Output range: [{cnn_output.min().item():.4f}, {cnn_output.max().item():.4f}]")

    # Test ResNet
    print("\n2️⃣  Testing GraspingResNet...")
    resnet_model = GraspingResNet(pretrained=False)  # Don't download pretrained weights during testing
    resnet_output = resnet_model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {resnet_output.shape}")
    print(f"   Parameters:   {count_parameters(resnet_model):,}")
    print(f"   Output range: [{resnet_output.min().item():.4f}, {resnet_output.max().item():.4f}]")

    # Test VGG
    print("\n3️⃣  Testing GraspingVGG...")
    vgg_model = GraspingVGG(pretrained=False)
    vgg_output = vgg_model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {vgg_output.shape}")
    print(f"   Parameters:   {count_parameters(vgg_model):,}")
    print(f"   Output range: [{vgg_output.min().item():.4f}, {vgg_output.max().item():.4f}]")

    print("\n" + "=" * 70)
    print("✅ All models tested successfully!")
    print("=" * 70)

    # Model comparison
    print("\n📊 Model Comparison:")
    print(f"   CNN:    {count_parameters(cnn_model)/1e6:.2f}M parameters")
    print(f"   ResNet: {count_parameters(resnet_model)/1e6:.2f}M parameters")
    print(f"   VGG:    {count_parameters(vgg_model)/1e6:.2f}M parameters")