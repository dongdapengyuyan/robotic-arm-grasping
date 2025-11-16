"""
Cornell Grasping Dataset Downloader
Downloads dataset directly to project directory
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path


def check_disk_space(path, required_gb=15):
    """Check if enough disk space available"""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)

    print(f"💾 Disk space check:")
    print(f"   Free space: {free_gb:.1f} GB")
    print(f"   Required:   {required_gb} GB")

    if free_gb < required_gb:
        print(f"\n❌ Insufficient disk space!")
        print(f"   Need at least {required_gb} GB free")
        print(f"   Please free up space and try again")
        return False

    print(f"   ✅ Sufficient space available\n")
    return True


def download_with_kagglehub():
    """Download using kagglehub to project directory"""

    try:
        import kagglehub
        print("✅ kagglehub is installed\n")
    except ImportError:
        print("❌ kagglehub not found! Installing...")
        os.system("pip install kagglehub")
        import kagglehub
        print("✅ kagglehub installed\n")

    # Set download path to project directory
    project_dir = Path.cwd()
    cache_dir = project_dir / ".kaggle_cache"
    cache_dir.mkdir(exist_ok=True)

    # Override kagglehub cache directory
    os.environ['KAGGLE_DATA_DIR'] = str(cache_dir)

    print("=" * 70)
    print("🔄 Downloading from Kaggle...")
    print("=" * 70)
    print(f"📂 Download location: {cache_dir}\n")

    try:
        # Download dataset
        download_path = kagglehub.dataset_download("oneoneliu/cornell-grasp")
        print(f"\n✅ Downloaded to: {download_path}")
        return download_path

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None


def download_with_kaggle_cli():
    """Download using Kaggle CLI to project directory"""

    print("=" * 70)
    print("🔄 Downloading with Kaggle CLI...")
    print("=" * 70)
    print()

    project_dir = Path.cwd()
    download_dir = project_dir / "downloads"
    download_dir.mkdir(exist_ok=True)

    zip_file = download_dir / "cornell-grasp.zip"

    # Download using kaggle CLI
    cmd = f'kaggle datasets download -d oneoneliu/cornell-grasp -p "{download_dir}"'

    print(f"📥 Executing: {cmd}\n")
    result = os.system(cmd)

    if result == 0 and zip_file.exists():
        print(f"\n✅ Downloaded to: {zip_file}")
        return str(zip_file)
    else:
        print(f"\n❌ Download failed")
        return None


def extract_dataset(zip_path, target_dir):
    """Extract ZIP file to target directory"""

    print("\n" + "=" * 70)
    print("📦 Extracting dataset...")
    print("=" * 70)

    zip_path = Path(zip_path)
    target_path = Path(target_dir)

    if not zip_path.exists():
        print(f"❌ ZIP file not found: {zip_path}")
        return False

    print(f"📂 From: {zip_path}")
    print(f"📂 To:   {target_path}")
    print()

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size
            total_size = sum(f.file_size for f in zip_ref.filelist)
            extracted_size = 0

            print(f"Total size: {total_size / (1024 ** 3):.2f} GB")
            print("Extracting files...")

            # Extract with progress
            for file in zip_ref.filelist:
                zip_ref.extract(file, target_path.parent)
                extracted_size += file.file_size

                if extracted_size % (100 * 1024 * 1024) == 0:  # Every 100MB
                    progress = (extracted_size / total_size) * 100
                    print(f"  Progress: {progress:.1f}%", end='\r')

            print(f"  Progress: 100.0%")

        print("\n✅ Extraction complete!")
        return True

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False


def organize_dataset(extract_path, target_path):
    """Organize dataset to correct structure"""

    extract_path = Path(extract_path)
    target_path = Path(target_path)

    # Find actual data directory
    possible_paths = [
        extract_path / "01",
        extract_path,
    ]

    source_dir = None
    for p in possible_paths:
        if p.exists() and any(p.glob("**/*.png")):
            source_dir = p
            break

    if source_dir is None:
        print("⚠️  Could not find dataset files")
        return False

    if source_dir != target_path:
        print(f"\n📦 Organizing dataset structure...")
        print(f"   From: {source_dir}")
        print(f"   To:   {target_path}")

        if target_path.exists():
            shutil.rmtree(target_path)

        shutil.copytree(source_dir, target_path)
        print("✅ Dataset organized\n")

    return True


def validate_dataset(dataset_path):
    """Validate dataset contents"""

    print("=" * 70)
    print("🔍 Validating dataset...")
    print("=" * 70)

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False

    png_files = list(dataset_path.glob("**/*.png"))
    txt_files = list(dataset_path.glob("**/*cpos.txt"))
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    print(f"✅ Found {len(png_files)} PNG images")
    print(f"✅ Found {len(txt_files)} grasp annotations")
    print(f"📁 Found {len(subdirs)} object directories")

    if len(subdirs) > 0:
        print(f"📂 Sample dirs: {[d.name for d in subdirs[:5]]}")

    print()

    if len(png_files) > 0 and len(txt_files) > 0:
        print("✅ Dataset validation passed!")
        return True
    else:
        print("❌ Dataset incomplete!")
        return False


def main():
    print("\n" + "=" * 70)
    print("🤖 Cornell Grasping Dataset Downloader")
    print("=" * 70)
    print(f"📁 Project directory: {Path.cwd()}\n")

    # Target directory in project
    target_dir = "cornell_dataset/datasets/oneoneliu/cornell-grasp/versions/1/01"
    target_path = Path(target_dir)

    # Check if dataset already exists
    if target_path.exists():
        png_files = list(target_path.glob("**/*.png"))
        if len(png_files) > 0:
            print(f"✅ Dataset already exists with {len(png_files)} images")
            response = input("\nRe-download? (y/n): ").lower()
            if response != 'y':
                print("✅ Using existing dataset")
                return
            shutil.rmtree("cornell_dataset", ignore_errors=True)

    # Check disk space (need ~15GB for download + extraction)
    if not check_disk_space(Path.cwd(), required_gb=15):
        print("\n💡 Suggestions:")
        print("   1. Free up disk space")
        print("   2. Move project to a drive with more space")
        print("   3. Manually download from Kaggle website")
        sys.exit(1)

    print("=" * 70)
    print("📥 Starting Download")
    print("=" * 70)
    print()

    # Try downloading with kagglehub first
    print("Method 1: Using kagglehub\n")
    download_path = download_with_kagglehub()

    # If kagglehub fails, try Kaggle CLI
    if download_path is None:
        print("\nMethod 2: Using Kaggle CLI\n")
        download_path = download_with_kaggle_cli()

    if download_path is None:
        print("\n" + "=" * 70)
        print("❌ Automatic download failed")
        print("=" * 70)
        print("\n💡 Manual Download:")
        print("   1. Visit: https://www.kaggle.com/datasets/oneoneliu/cornell-grasp")
        print("   2. Click 'Download' button")
        print(f"   3. Save ZIP to: {Path.cwd() / 'downloads'}")
        print("   4. Run this script again to extract")
        sys.exit(1)

    # If downloaded file is a ZIP, extract it
    download_path = Path(download_path)

    if download_path.suffix == '.zip':
        if extract_dataset(download_path, target_path):
            validate_dataset(target_path)
        else:
            sys.exit(1)
    else:
        # kagglehub already extracted, just organize
        if organize_dataset(download_path, target_path):
            validate_dataset(target_path)
        else:
            sys.exit(1)

    print("\n" + "=" * 70)
    print("🎉 SUCCESS!")
    print("=" * 70)
    print(f"📂 Dataset location: {target_path.absolute()}")
    print("\nNext step: python train_models.py\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        import os
        import sys
        import time
        from pathlib import Path
        import shutil


        def download_cornell_dataset():
            """Download Cornell Grasping dataset using kagglehub"""

            print("=" * 70)
            print("📥 Cornell Grasping Dataset Downloader")
            print("=" * 70)
            print()

            # Check if kagglehub is installed
            try:
                import kagglehub
                print("✅ kagglehub is installed")
            except ImportError:
                print("❌ kagglehub not found!")
                print()
                print("📦 Installing kagglehub...")
                os.system("pip install kagglehub")
                print()
                try:
                    import kagglehub
                    print("✅ kagglehub installed successfully")
                except:
                    print("❌ Failed to install kagglehub")
                    print("💡 Please run manually: pip install kagglehub")
                    sys.exit(1)

            print()

            # Target path (consistent with the path in train_models.py)
            target_dir = "cornell_dataset/datasets/oneoneliu/cornell-grasp/versions/1/01"
            target_path = Path(target_dir)

            # Check if dataset already exists
            if target_path.exists():
                print(f"⚠️  Dataset already exists at: {target_path}")
                print()
                response = input("Do you want to re-download? (y/n): ").lower()
                if response != 'y':
                    print("✅ Using existing dataset")
                    print(f"📂 Dataset location: {target_path.absolute()}")
                    return str(target_path.absolute())
                else:
                    print("🗑️  Removing existing dataset...")
                    shutil.rmtree("cornell_dataset", ignore_errors=True)

            print()
            print("=" * 70)
            print("🔄 Starting download from Kaggle...")
            print("=" * 70)
            print()

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    print(f"📡 Attempt {retry_count + 1}/{max_retries}")
                    print("⏳ This may take several minutes depending on your connection...")
                    print()

                    # Set environment variable (increase timeout)
                    os.environ['KAGGLE_TIMEOUT'] = '600'  # 10 minute timeout

                    # Download dataset
                    download_path = kagglehub.dataset_download("oneoneliu/cornell-grasp")

                    print()
                    print(f"✅ Download completed!")
                    print(f"📂 Downloaded to: {download_path}")
                    print()

                    # Check downloaded file structure
                    download_path_obj = Path(download_path)

                    # Possible subdirectory structures
                    possible_paths = [
                        download_path_obj / "datasets/oneoneliu/cornell-grasp/versions/1/01",
                        download_path_obj / "01",
                        download_path_obj,
                    ]

                    source_dir = None
                    for p in possible_paths:
                        if p.exists() and any(p.iterdir()):
                            source_dir = p
                            break

                    if source_dir is None:
                        print("⚠️  Unexpected directory structure")
                        print(f"📂 Download location: {download_path}")
                        print("💡 Please check the directory and manually move files to:")
                        print(f"   {target_path.absolute()}")
                        return download_path

                    # Create target directory
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move files to standard location
                    if source_dir != target_path:
                        print(f"📦 Organizing dataset structure...")
                        print(f"   From: {source_dir}")
                        print(f"   To:   {target_path}")

                        if target_path.exists():
                            shutil.rmtree(target_path)

                        shutil.copytree(source_dir, target_path)
                        print("✅ Dataset organized successfully")

                    print()
                    print("=" * 70)
                    print("✅ DOWNLOAD COMPLETE!")
                    print("=" * 70)
                    print()
                    print(f"📂 Dataset location: {target_path.absolute()}")
                    print(f"📊 Ready for training!")
                    print()

                    # Validate dataset content
                    validate_dataset(target_path)

                    return str(target_path.absolute())

                except Exception as e:
                    retry_count += 1
                    print(f"❌ Attempt {retry_count} failed: {e}")
                    print()

                    if retry_count < max_retries:
                        wait_time = 10 * retry_count
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        print()
                    else:
                        print()
                        print("=" * 70)
                        print("❌ Download failed after multiple attempts")
                        print("=" * 70)
                        print()
                        print_manual_download_instructions()
                        sys.exit(1)


        def validate_dataset(dataset_path):
            """Validate dataset content"""
            print("=" * 70)
            print("🔍 Validating dataset...")
            print("=" * 70)

            dataset_path = Path(dataset_path)

            # Count files
            png_files = list(dataset_path.glob("**/*.png"))
            txt_files = list(dataset_path.glob("**/*cpos.txt"))

            print(f"✅ Found {len(png_files)} PNG images")
            print(f"✅ Found {len(txt_files)} grasp annotation files")

            if len(png_files) == 0:
                print("⚠️  Warning: No PNG files found!")
                return False

            if len(txt_files) == 0:
                print("⚠️  Warning: No annotation files found!")
                return False

            # Check subdirectories
            subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            print(f"📁 Found {len(subdirs)} object directories")

            if len(subdirs) > 0:
                print(f"📂 Sample directories: {[d.name for d in subdirs[:5]]}")

            print()
            return True


        def print_manual_download_instructions():
            """Print manual download instructions"""
            print("💡 MANUAL DOWNLOAD INSTRUCTIONS")
            print("=" * 70)
            print()
            print("Method 1: Download from Kaggle Website")
            print("-" * 70)
            print("1. Visit: https://www.kaggle.com/datasets/oneoneliu/cornell-grasp")
            print("2. Click the 'Download' button (requires Kaggle account)")
            print("3. Extract the downloaded ZIP file")
            print("4. Move/copy the contents to:")
            print(f"   cornell_dataset/datasets/oneoneliu/cornell-grasp/versions/1/01/")
            print()
            print("Method 2: Use Kaggle CLI")
            print("-" * 70)
            print("1. Install Kaggle CLI:")
            print("   pip install kaggle")
            print()
            print("2. Set up Kaggle API credentials:")
            print("   - Go to https://www.kaggle.com/settings")
            print("   - Click 'Create New API Token'")
            print("   - Save kaggle.json to:")
            print("     Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
            print("     Linux/Mac: ~/.kaggle/kaggle.json")
            print()
            print("3. Download dataset:")
            print("   kaggle datasets download -d oneoneliu/cornell-grasp")
            print()
            print("4. Extract and organize:")
            print("   unzip cornell-grasp.zip -d cornell_dataset/")
            print()
            print("Method 3: Use a VPN or Proxy")
            print("-" * 70)
            print("If you're experiencing network issues:")
            print("- Try using a VPN")
            print("- Configure a proxy")
            print("- Try downloading at a different time")
            print()
            print("=" * 70)


        def check_dataset_ready():
            """Check if dataset is available"""
            target_dir = "cornell_dataset/datasets/oneoneliu/cornell-grasp/versions/1/01"
            target_path = Path(target_dir)

            if target_path.exists():
                png_files = list(target_path.glob("**/*.png"))
                if len(png_files) > 0:
                    print("✅ Dataset is ready for training!")
                    print(f"📂 Location: {target_path.absolute()}")
                    print(f"📊 Found {len(png_files)} images")
                    return True

            print("❌ Dataset not found or incomplete")
            return False


        def main():
            print()
            print("🤖 Cornell Grasping Dataset Downloader")
            print("=" * 70)
            print()

            # Display current working directory
            print(f"📁 Current directory: {Path.cwd()}")
            print()

            # Check if dataset already exists
            if check_dataset_ready():
                print()
                response = input("Dataset already exists. Download again? (y/n): ").lower()
                if response != 'y':
                    print("✅ Using existing dataset")
                    return

            print()

            # Start download
            try:
                dataset_path = download_cornell_dataset()
                print()
                print("🎉 SUCCESS!")
                print()
                print("Next steps:")
                print("1. Verify the dataset structure")
                print("2. Run training script:")
                print("   python train_models.py")
                print()

            except KeyboardInterrupt:
                print()
                print("❌ Download cancelled by user")
                sys.exit(1)
            except Exception as e:
                print()
                print(f"❌ Unexpected error: {e}")
                print()
                print_manual_download_instructions()
                sys.exit(1)


        if __name__ == '__main__':
            main()
        traceback.print_exc()
        sys.exit(1)