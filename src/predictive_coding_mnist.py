import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001 # Placeholder, PC might need different tuning
EPOCHS = 10 # Placeholder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- MNIST Data Loading ---
def get_mnist_loaders(batch_size):
    """
    Returns DataLoader instances for MNIST train and test sets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image or numpy.ndarray to tensor and scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific mean and std
    ])

    # Ensure the data directory exists within the project structure
    data_dir = './data' # Changed from root='./data' to be relative

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 # Added for potentially faster loading
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 # Added for potentially faster loading
    )
    print(f"MNIST training data loaded: {len(train_dataset)} samples")
    print(f"MNIST test data loaded: {len(test_dataset)} samples")
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    # Example of iterating through the data
    print("\n--- Example Batch ---")
    try:
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print(f"Images batch shape: {images.shape}") # Should be [BATCH_SIZE, 1, 28, 28]
        print(f"Labels batch shape: {labels.shape}") # Should be [BATCH_SIZE]

        # Flatten images for an MLP-style input
        images_flattened = images.view(images.shape[0], -1)
        print(f"Flattened images batch shape: {images_flattened.shape}") # Should be [BATCH_SIZE, 784]

        print("\nSetup complete. Next steps would be defining the Predictive Coding Network.")
    except Exception as e:
        print(f"\nError getting example batch: {e}")
        print("This might happen if the script is run in an environment where data downloading or multiprocessing is restricted.")
