
import os
import glob

dataset_dir = r"D:\2025fighting\69_CSCI323_MSK\robotic-arm-grasping\cornell_dataset\datasets\oneoneliu\cornell-grasp\versions\1\01"

# Find all annotation files
grasp_files = glob.glob(os.path.join(dataset_dir, "**", "*cpos.txt"), recursive=True)

print(f"Found {len(grasp_files)} grasp annotation files\n")

# Check the content of the first 5 files
for i, filepath in enumerate(grasp_files[:5]):
    print(f"{'=' * 60}")
    print(f"File {i + 1}: {os.path.basename(filepath)}")
    print(f"{'=' * 60}")

    with open(filepath, 'r') as f:
        content = f.read()
        print(content)
    print()

# Calculate coordinate ranges
print(f"\n{'=' * 60}")
print("Analyzing coordinate ranges...")
print(f"{'=' * 60}")

all_coords = []
for filepath in grasp_files:
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if lines:
            coords = list(map(float, lines[0].strip().split()))
            all_coords.append(coords)

if all_coords:
    import numpy as np

    all_coords = np.array(all_coords)

    print(f"Total samples: {len(all_coords)}")
    print(f"Coordinates per sample: {all_coords.shape[1] if len(all_coords) > 0 else 0}")
    print(f"\nCoordinate ranges:")
    for i in range(min(8, all_coords.shape[1])):
        print(f"  Coord[{i}]: [{all_coords[:, i].min():.2f}, {all_coords[:, i].max():.2f}]")