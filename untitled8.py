# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define the CNN model class
class FastFashionNet(nn.Module):
    def __init__(self):
        super(FastFashionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
 # Define the forward pass
    def forward(self, x):
        return self.network(x)
# Function to train the model and save its weights
def train_and_save():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

      # Set up data transformation and load training data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.2860,), (0.3530,))  # Normalize with mean and std specific to Fashion MNIST
    ])
    ])

    train_data = datasets.FashionMNIST(
        root='./data',   # Directory to store data
        train=True,      # Load training data
        transform=transform, # Apply transformations
        download=True    # Download if data isn't already available
    )
  # Create a DataLoader to handle batching of data
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = FastFashionNet().to(device)
    criterion = nn.CrossEntropyLoss()         # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer for efficient training

   # Train the model for a set number of epochs
    print("Training started...")
    for epoch in range(3): #Run for 3 epochs as a demonstration
        model.train()   # Set model to training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) # Move data and labels to the chosen device
            optimizer.zero_grad()        # Reset gradients to zero
            output = model(data)          # Forward pass through the model
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()            # Backpropagate to compute gradients
            optimizer.step()             # Update model weights

            # Print loss periodically to monitor training progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}: [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

     # Save the model weights after training
    torch.save(model.state_dict(), 'fashion_mnist_weights.pth')
    print("Training completed and weights saved!")
    return model
# Function to evaluate the model's performance on test data
def evaluate_model(model=None):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model weights if a model is not passed in
    if model is None:
        model = FastFashionNet().to(device)
        model.load_state_dict(torch.load("fashion_mnist_weights.pth", map_location=device))

    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Load test data with the same transformations as training data
    test_data = datasets.FashionMNIST(
        root='./data',
        train=False,            # Load test data
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        download=True
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Variables to track total and correct predictions for accuracy calculation
    correct = 0
    total = 0
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    # Class names for Fashion MNIST
    # Disable gradient calculations for faster evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # Move data to device
            outputs = model(images)                                # Forward pass
            _, predicted = outputs.max(1)                            # Get class predictions
            total += labels.size(0)                                 # Count total labels
            correct += predicted.eq(labels).sum().item()         # Count correct predictions
 # Calculate accuracy as percentage of correct predictions
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Run training and evaluation if this file is run as a script
if __name__ == "__main__":
     # Train model and save weights
    model = train_and_save()

   # Evaluate the model on test data
    evaluate_model(model)
