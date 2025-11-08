"""
修复后的Cornell数据集加载器
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CornellGraspDataset(Dataset):
    """Cornell抓取数据集 - 修复版"""

    def __init__(self, root_dir, split='train', train_ratio=0.7, val_ratio=0.15):
        self.root_dir = root_dir
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 查找所有图片和标注
        self.samples = self._load_dataset()

        # 划分数据集
        total_samples = len(self.samples)
        np.random.seed(42)
        indices = np.random.permutation(total_samples)

        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]

        print(f"  {split.capitalize()}: {len(self.indices)} samples")

    def _load_dataset(self):
        """加载所有图片和对应的抓取标注"""
        samples = []

        # 查找所有RGB图片
        rgb_files = glob.glob(os.path.join(self.root_dir, "**", "pcd*r.png"),
                              recursive=True)

        print(f"  Found {len(rgb_files)} RGB images")

        for rgb_path in rgb_files:
            # 构建对应的标注文件路径
            base_name = os.path.basename(rgb_path).replace('r.png', '')
            grasp_path = rgb_path.replace('r.png', 'cpos.txt')

            if os.path.exists(grasp_path):
                # 解析抓取框
                grasps = self._parse_grasp_file(grasp_path)
                if len(grasps) > 0:
                    samples.append({
                        'image_path': rgb_path,
                        'grasps': grasps
                    })

        print(f"  Loaded {len(samples)} samples with annotations")
        return samples

    def _parse_grasp_file(self, filepath):
        """
        解析Cornell抓取标注文件
        每个文件包含多个抓取框，每个框由4个点定义
        """
        grasps = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 每4行定义一个抓取框
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                try:
                    # 解析4个点的坐标
                    points = []
                    for j in range(4):
                        coords = list(map(float, lines[i + j].strip().split()))
                        if len(coords) == 2:
                            points.append(coords)

                    if len(points) == 4:
                        # 计算抓取中心和角度
                        center, angle, width, height = self._compute_grasp_params(points)

                        grasps.append({
                            'center': center,
                            'angle': angle,
                            'width': width,
                            'height': height,
                            'points': points
                        })
                except:
                    continue

        return grasps

    def _compute_grasp_params(self, points):
        """
        从4个点计算抓取参数
        points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        返回: center, angle, width, height
        """
        points = np.array(points)

        # 计算中心点
        center = points.mean(axis=0)

        # 计算主轴方向（连接对角点）
        # 假设点的顺序是逆时针或顺时针
        edge1 = points[1] - points[0]  # 第一条边
        edge2 = points[2] - points[1]  # 第二条边

        # 选择较长的边作为抓取方向
        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)

        if len1 > len2:
            grasp_direction = edge1
            width = len1
            height = len2
        else:
            grasp_direction = edge2
            width = len2
            height = len1

        # 计算角度 (弧度转角度)
        angle = np.arctan2(grasp_direction[1], grasp_direction[0])
        angle = np.degrees(angle)

        # 归一化角度到 [0, 180]
        angle = angle % 180

        return center, angle, width, height

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.samples[actual_idx]

        # 加载图片
        image = Image.open(sample['image_path']).convert('RGB')
        orig_width, orig_height = image.size

        # 随机选择一个抓取框
        grasp = np.random.choice(sample['grasps'])

        # 归一化坐标到 [0, 1]
        center_x = grasp['center'][0] / orig_width
        center_y = grasp['center'][1] / orig_height

        # 归一化角度到 [0, 1]
        angle_normalized = grasp['angle'] / 180.0

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'center_x': torch.tensor(center_x, dtype=torch.float32),
            'center_y': torch.tensor(center_y, dtype=torch.float32),
            'angle': torch.tensor(angle_normalized, dtype=torch.float32)
        }


def get_cornell_dataloaders(root_dir, batch_size=32, num_workers=0):
    """创建Cornell数据集的DataLoader"""

    print(f"📂 Loading Cornell dataset from: {root_dir}")

    train_dataset = CornellGraspDataset(root_dir, split='train')
    val_dataset = CornellGraspDataset(root_dir, split='val')
    test_dataset = CornellGraspDataset(root_dir, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == '__main__':
    root_dir = r"D:\2025fighting\69_CSCI323_MSK\robotic-arm-grasping\cornell_dataset\datasets\oneoneliu\cornell-grasp\versions\1\01"

    print("=" * 60)
    print("Testing Fixed Cornell Dataset Loader")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_cornell_dataloaders(
        root_dir,
        batch_size=4
    )

    # 测试一个batch
    print("\n" + "=" * 60)
    print("Testing first batch")
    print("=" * 60)

    batch = next(iter(train_loader))

    print(f"Image shape: {batch['image'].shape}")
    print(f"Center X range: [{batch['center_x'].min():.3f}, {batch['center_x'].max():.3f}]")
    print(f"Center Y range: [{batch['center_y'].min():.3f}, {batch['center_y'].max():.3f}]")
    print(f"Angle range: [{batch['angle'].min():.3f}, {batch['angle'].max():.3f}]")
    print(f"Angle (degrees): [{batch['angle'].min() * 180:.1f}°, {batch['angle'].max() * 180:.1f}°]")