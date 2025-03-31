import subprocess
import sys
import os

def install_requirements():
    """Install requirements using pip with Tsinghua mirror."""
    print("Installing requirements from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "/root/code/louye/requirements.txt",
            "-i", 
            "https://pypi.tuna.tsinghua.edu.cn/simple"
        ])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def run_training():
    """Run the training script."""
    print("Starting training...")
    try:
        subprocess.check_call([sys.executable, "train.py"])
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def main():
    """Main function to install requirements and run training."""
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found!")
        sys.exit(1)

    # Install requirements
    install_requirements()

    # Run training
    run_training()

if __name__ == "__main__":
    main() 