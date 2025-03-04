import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

def generate_parabola_data(coefficients = [1, 2, 3, 4], num_points=20, noise=0.1, x_range=(-2, 2)):

    import numpy as np
    import torch
    
    np.random.seed(42)
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    y_true = np.zeros_like(x)
    for i, coef in enumerate(coefficients):
        y_true += coef * (x ** i)
    
    y = y_true + np.random.normal(0, noise, size=num_points)
    
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    dataset = list(zip(x_tensor, y_tensor))
    
    return dataset, x, y, y_true

class StandardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StandardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Standard ReLU
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x, self.weight, self.bias

class TunableReLU(nn.Module):
    def __init__(self, hidden_size=16, epsilon_init=0.1):
        super(TunableReLU, self).__init__()
        self.epsilon = nn.Parameter(torch.full((hidden_size,), epsilon_init, dtype=torch.float32))
        self.epsilon_history = []
    
    def forward(self, x, weight, bias):

        batch_size = x.size(0)
        x_expanded = x.expand(batch_size, weight.size(0))
        
        output = x_expanded * weight.t()
        
        positive_sum = torch.where(output > 0, output, torch.zeros_like(output))
        negative_sum = torch.where(output < 0, output, torch.zeros_like(output))
        
        bias_expanded = bias.unsqueeze(0).expand(batch_size, -1)
        mu = positive_sum + torch.where(bias_expanded >= 0, bias_expanded, torch.zeros_like(bias_expanded))
        nu = negative_sum + torch.where(bias_expanded < 0, bias_expanded, torch.zeros_like(bias_expanded))
        
        epsilon_expanded = self.epsilon.unsqueeze(0).expand(batch_size, -1)
        temp = mu - nu - epsilon_expanded
        term = temp ** 2 + 4 * mu * epsilon_expanded
        activation = temp + torch.sqrt(term)
        
        self.epsilon_history.append(self.epsilon.mean().item())
        
        return activation

class TunableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, epsilon):
        super(TunableNN, self).__init__()
        self.fc1 = CustomLinear(input_size, hidden_size)
        self.tunable_relu = TunableReLU(hidden_size=hidden_size, epsilon_init=epsilon)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, weights, bias1 = self.fc1(x)
        x = self.tunable_relu(x, weights, bias1)
        x = self.fc2(x)
        return x
    
    def get_epsilon_history(self):
        return self.tunable_relu.epsilon_history

def train_model(model, dataset, criterion, optimizer, epochs=100):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for x_sample, y_sample in dataset:
            optimizer.zero_grad()
            output = model(x_sample)
            loss = criterion(output, y_sample)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def generate_predictions(model, x_range=(-2.5, 2.5), num_points=100):
    model.eval()
    x_pred = np.linspace(x_range[0], x_range[1], num_points)
    x_pred_tensor = torch.tensor(x_pred, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        y_pred = model(x_pred_tensor).numpy().flatten()
    
    return x_pred, y_pred

# -------------------------------
# Main execution
# -------------------------------

num_points = 20
dataset, x_data, y_data, y_true = generate_parabola_data(num_points=num_points, noise=0.15)

# Hyperparameters
input_size = 1     # x value
hidden_size = 10
output_size = 1    # y value (parabola height)
learning_rate = 0.01
epsilon = 0.1
epochs = 400

# Initialize models
standard_model = StandardNN(input_size, hidden_size, output_size)
tunable_model = TunableNN(input_size, hidden_size, output_size, epsilon)

# Define loss and optimizers
criterion = nn.MSELoss()
standard_optimizer = optim.Adam(standard_model.parameters(), lr=learning_rate)
tunable_optimizer = optim.Adam(tunable_model.parameters(), lr=learning_rate)

# Train standard model
print("Training Standard ReLU model...")
standard_losses = train_model(standard_model, dataset, criterion, standard_optimizer, epochs=epochs)

# Train tunable model
print("\nTraining Tunable ReLU model...")
tunable_losses = train_model(tunable_model, dataset, criterion, tunable_optimizer, epochs=epochs)

# Get epsilon history
epsilon_history = tunable_model.get_epsilon_history()

# Generate predictions for visualization
x_pred, standard_pred = generate_predictions(standard_model)
_, tunable_pred = generate_predictions(tunable_model)

# Calculate true parabola for the prediction range
y_true_pred = x_pred**2

# Calculate final MSE for both models
with torch.no_grad():
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
    
    standard_output = standard_model(x_tensor)
    standard_mse = criterion(standard_output, y_tensor).item()
    
    tunable_output = tunable_model(x_tensor)
    tunable_mse = criterion(tunable_output, y_tensor).item()

print(f"\nFinal MSE - Standard ReLU: {standard_mse:.6f}, Tunable ReLU: {tunable_mse:.6f}")

# -------------------------------
# Visualizations
# -------------------------------
# Plot the training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(standard_losses, label='Standard ReLU')
plt.plot(tunable_losses, label='Tunable ReLU')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# Plot epsilon evolution
plt.subplot(1, 2, 2)
plt.plot(epsilon_history)
plt.xlabel('Training Step')
plt.ylabel('Epsilon Value (Mean)')
plt.title('Epsilon Parameter Evolution')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the fitted curves
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)

# Plot data points and true parabola
plt.scatter(x_data, y_data, color='blue', label='Data Points')
plt.plot(x_pred, y_true_pred, 'g--', label='True Parabola y=x²')
plt.plot(x_pred, standard_pred, 'r-', label=f'Standard ReLU (MSE: {standard_mse:.6f})')
plt.plot(x_pred, tunable_pred, 'm-', label=f'Tunable ReLU (MSE: {tunable_mse:.6f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parabola Fitting Comparison')
plt.legend()
plt.grid(True)

# Plot the residuals (difference between predictions and true values)
plt.subplot(2, 1, 2)
plt.plot(x_pred, standard_pred - y_true_pred, 'r-', label='Standard ReLU Error')
plt.plot(x_pred, tunable_pred - y_true_pred, 'm-', label='Tunable ReLU Error')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('x')
plt.ylabel('Residual (Prediction - True)')
plt.title('Error Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Try with different hidden layer sizes to see how model capacity affects fitting
hidden_sizes = [3, 5, 10, 20]
plt.figure(figsize=(15, 10))

for i, hs in enumerate(hidden_sizes):
    # Initialize models with this hidden size
    std_model = StandardNN(input_size, hs, output_size)
    tun_model = TunableNN(input_size, hs, output_size, epsilon)
    
    # Define optimizers
    std_optimizer = optim.Adam(std_model.parameters(), lr=learning_rate)
    tun_optimizer = optim.Adam(tun_model.parameters(), lr=learning_rate)
    
    # Train both models
    train_model(std_model, dataset, criterion, std_optimizer, epochs=epochs)
    train_model(tun_model, dataset, criterion, tun_optimizer, epochs=epochs)
    
    # Generate predictions
    x_pred, std_pred = generate_predictions(std_model)
    _, tun_pred = generate_predictions(tun_model)
    
    # Calculate MSE
    with torch.no_grad():
        std_output = std_model(torch.tensor(x_data, dtype=torch.float32).view(-1, 1))
        std_mse = criterion(std_output, torch.tensor(y_data, dtype=torch.float32).view(-1, 1)).item()
        
        tun_output = tun_model(torch.tensor(x_data, dtype=torch.float32).view(-1, 1))
        tun_mse = criterion(tun_output, torch.tensor(y_data, dtype=torch.float32).view(-1, 1)).item()
    
    # Plot
    plt.subplot(2, 2, i+1)
    plt.scatter(x_data, y_data, color='blue', label='Data Points')
    plt.plot(x_pred, y_true_pred, 'g--', label='True Parabola')
    plt.plot(x_pred, std_pred, 'r-', label=f'Standard (MSE: {std_mse:.4f})')
    plt.plot(x_pred, tun_pred, 'm-', label=f'Tunable (MSE: {tun_mse:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Hidden Size = {hs}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()