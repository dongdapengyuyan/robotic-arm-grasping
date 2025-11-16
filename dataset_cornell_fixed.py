"""
Fixed Cornell dataset loader
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CornellGraspDataset(Dataset):
    """Cornell grasp dataset - fixed version"""

    def __init__(self, root_dir, split='train', train_ratio=0.7, val_ratio=0.15):
        self.root_dir = root_dir
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Find all images and annotations
        self.samples = self._load_dataset()

        # Split dataset
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
        """Load all images and corresponding grasp annotations"""
        samples = []

        # Find all RGB images
        rgb_files = glob.glob(os.path.join(self.root_dir, "**", "pcd*r.png"),
                              recursive=True)

        print(f"  Found {len(rgb_files)} RGB images")

        for rgb_path in rgb_files:
            # Construct corresponding annotation file path
            base_name = os.path.basename(rgb_path).replace('r.png', '')
            grasp_path = rgb_path.replace('r.png', 'cpos.txt')

            if os.path.exists(grasp_path):
                # Parse grasp rectangles
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
        Parse Cornell grasp annotation file
        Each file contains multiple grasp rectangles, each defined by 4 points
        """
        grasps = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Every 4 lines define a grasp rectangle
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                try:
                    # Parse coordinates of 4 points
                    points = []
                    for j in range(4):
                        coords = list(map(float, lines[i + j].strip().split()))
                        if len(coords) == 2:
                            points.append(coords)

                    if len(points) == 4:
                        # Calculate grasp center and angle
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
        Calculate grasp parameters from 4 points
        points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns: center, angle, width, height
        """
        points = np.array(points)

        # Calculate center point
        center = points.mean(axis=0)

        # Calculate principal axis direction (connecting diagonal points)
        # Assume points are in counterclockwise or clockwise order
        edge1 = points[1] - points[0]  # First edge
        edge2 = points[2] - points[1]  # Second edge

        # Choose the longer edge as grasp direction
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

        # Calculate angle (radians to degrees)
        angle = np.arctan2(grasp_direction[1], grasp_direction[0])
        angle = np.degrees(angle)

        # Normalize angle to [0, 180]
        angle = angle % 180

        return center, angle, width, height

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.samples[actual_idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        orig_width, orig_height = image.size

        # Randomly select a grasp rectangle
        grasp = np.random.choice(sample['grasps'])

        # Normalize coordinates to [0, 1]
        center_x = grasp['center'][0] / orig_width
        center_y = grasp['center'][1] / orig_height

        # Normalize angle to [0, 1]
        angle_normalized = grasp['angle'] / 180.0

        # Apply image transform
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'center_x': torch.tensor(center_x, dtype=torch.float32),
            'center_y': torch.tensor(center_y, dtype=torch.float32),
            'angle': torch.tensor(angle_normalized, dtype=torch.float32)
        }


def get_cornell_dataloaders(root_dir, batch_size=32, num_workers=0):
    """Create DataLoaders for Cornell dataset"""

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


# Test code
if __name__ == '__main__':
    root_dir = r"D:\2025fighting\69_CSCI323_MSK\robotic-arm-grasping\cornell_dataset\datasets\oneoneliu\cornell-grasp\versions\1\01"

    print("=" * 60)
    print("Testing Fixed Cornell Dataset Loader")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_cornell_dataloaders(
        root_dir,
        batch_size=4
    )

    # Test one batch
    print("\n" + "=" * 60)
    print("Testing first batch")
    print("=" * 60)

    batch = next(iter(train_loader))

    print(f"Image shape: {batch['image'].shape}")
    print(f"Center X range: [{batch['center_x'].min():.3f}, {batch['center_x'].max():.3f}]")
    print(f"Center Y range: [{batch['center_y'].min():.3f}, {batch['center_y'].max():.3f}]")
    print(f"Angle range: [{batch['angle'].min():.3f}, {batch['angle'].max():.3f}]")
    print(f"Angle (degrees): [{batch['angle'].min() * 180:.1f}°, {batch['angle'].max() * 180:.1f}°]")