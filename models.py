"""
深度学习模型定义
"""
import torch
import torch.nn as nn
import torchvision.models as models


class GraspingCNN(nn.Module):
    """自定义CNN模型用于抓取预测"""

    def __init__(self, input_channels=3, output_dim=3):
        """
        Args:
            input_channels: 输入图像通道数 (RGB=3)
            output_dim: 输出维度 (x, y, angle) = 3
        """
        super(GraspingCNN, self).__init__()

        # 卷积层
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

        # 全连接层
        # 输入图像 224x224，经过4次maxpool (2x2)，特征图大小为 14x14
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
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
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


class GraspingResNet(nn.Module):
    """基于ResNet的抓取预测模型"""

    def __init__(self, output_dim=3, pretrained=True):
        """
        Args:
            output_dim: 输出维度 (x, y, angle) = 3
            pretrained: 是否使用预训练权重
        """
        super(GraspingResNet, self).__init__()

        # 加载预训练的ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # 获取ResNet的特征维度
        num_features = self.resnet.fc.in_features

        # 替换最后的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

    def forward(self, x):
        """前向传播"""
        return self.resnet(x)


class GraspingVGG(nn.Module):
    """基于VGG的抓取预测模型（可选）"""

    def __init__(self, output_dim=3, pretrained=True):
        """
        Args:
            output_dim: 输出维度 (x, y, angle) = 3
            pretrained: 是否使用预训练权重
        """
        super(GraspingVGG, self).__init__()

        # 加载预训练的VGG16
        self.vgg = models.vgg16(pretrained=pretrained)

        # 获取VGG的特征维度
        num_features = self.vgg.classifier[0].in_features

        # 替换分类器
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
        """前向传播"""
        return self.vgg(x)


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """测试模型"""

    print("=" * 70)
    print("               🧪 Testing Models")
    print("=" * 70)

    # 创建随机输入 (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)

    # 测试CNN
    print("\n1️⃣  Testing GraspingCNN...")
    cnn_model = GraspingCNN()
    cnn_output = cnn_model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {cnn_output.shape}")
    print(f"   Parameters:   {count_parameters(cnn_model):,}")
    print(f"   Output range: [{cnn_output.min().item():.4f}, {cnn_output.max().item():.4f}]")

    # 测试ResNet
    print("\n2️⃣  Testing GraspingResNet...")
    resnet_model = GraspingResNet(pretrained=False)  # 测试时不下载预训练权重
    resnet_output = resnet_model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {resnet_output.shape}")
    print(f"   Parameters:   {count_parameters(resnet_model):,}")
    print(f"   Output range: [{resnet_output.min().item():.4f}, {resnet_output.max().item():.4f}]")

    # 测试VGG
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

    # 模型对比
    print("\n📊 Model Comparison:")
    print(f"   CNN:    {count_parameters(cnn_model)/1e6:.2f}M parameters")
    print(f"   ResNet: {count_parameters(resnet_model)/1e6:.2f}M parameters")
    print(f"   VGG:    {count_parameters(vgg_model)/1e6:.2f}M parameters")