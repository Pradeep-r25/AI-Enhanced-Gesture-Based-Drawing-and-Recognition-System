import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from sklearn.model_selection import train_test_split


# Define the model
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        # Input: 21 landmarks x 2 coordinates (x, y)
        self.fc1 = nn.Linear(21 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 classes: no gesture, fist, thumbs down

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    try:
        with open('gesture_dataset/gesture_data.json', 'r') as f:
            dataset = json.load(f)
        print(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        print("Dataset file not found. Please run collect_gesture_data.py first.")
        return

    # Extract features and labels
    X = [sample['landmarks'] for sample in dataset]
    y = [sample['label'] for sample in dataset]

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize the model
    model = GestureRecognitionModel().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 200
    batch_size = 32

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Process in batches
        for i in range(0, X_train.shape[0], batch_size):
            # Get batch
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test).sum().item() / y_test.size(0)

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

        # Calculate per-class accuracy
        class_names = ["no gesture", "fist", "thumbs down"]
        for i in range(3):
            class_mask = (y_test == i)
            if class_mask.sum().item() > 0:
                class_acc = (predicted[class_mask] == i).sum().item() / class_mask.sum().item()
                print(f"Accuracy for class '{class_names[i]}': {class_acc:.4f}")

        print(f"Overall test accuracy: {accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'gesture_model.pth')
    print("Model trained and saved as 'gesture_model.pth'")


if __name__ == "__main__":
    main()
