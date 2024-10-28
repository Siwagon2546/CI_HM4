import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations = [X]
        for W in self.weights[:-1]:
            X = self.sigmoid(np.dot(X, W))
            self.activations.append(X)
        output = np.dot(X, self.weights[-1])  # Last layer (linear)
        self.activations.append(output)
        return output

    def loss(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))  # Mean Absolute Error (MAE)

class Particle:
    def __init__(self, layer_sizes):
        self.position = MLP(layer_sizes)
        self.velocity = [np.zeros_like(W) for W in self.position.weights]
        self.best_position = self.position
        self.best_loss = float('inf')

def pso_train(X, y, layer_sizes, num_particles=5, max_iter=100, inertia=1, cognitive_const=2, social_const=2):
    particles = [Particle(layer_sizes) for _ in range(num_particles)]
    global_best_position = particles[0].position
    global_best_loss = float('inf')
    
    mse_progress = []  # Track MSE progress per iteration

    for i in range(max_iter):
        for particle in particles:
            # Forward pass
            y_pred = particle.position.forward(X)
            current_loss = particle.position.loss(y_pred, y)
            
            # Update personal best
            if current_loss < particle.best_loss:
                particle.best_position = particle.position
                particle.best_loss = current_loss
            
            # Update global best
            if current_loss < global_best_loss:
                global_best_position = particle.position
                global_best_loss = current_loss
            
            # Update velocity and position for each layer
            for l in range(len(particle.velocity)):
                cognitive_velocity = cognitive_const * np.random.rand() * (particle.best_position.weights[l] - particle.position.weights[l])
                social_velocity = social_const * np.random.rand() * (global_best_position.weights[l] - particle.position.weights[l])
                particle.velocity[l] = inertia * particle.velocity[l] + cognitive_velocity + social_velocity
                
                # Update position
                particle.position.weights[l] += particle.velocity[l]
            
        # Store MSE of the best particle in this iteration
        mse_progress.append(global_best_loss)
        
        if i % 10 == 0:
            print(f'Iteration {i}, MSE: {global_best_loss}')
    
    return global_best_position, global_best_loss, mse_progress

def manual_k_fold_split(X, y, k=10):
    fold_size = len(X) // k
    folds = []
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        folds.append((X_train, y_train, X_val, y_val))
    
    return folds

def k_fold_cross_validation(X, y, layer_sizes, num_particles=50, max_iter=100, inertia=0.5, cognitive_const=2, social_const=2, k=10):
    folds = manual_k_fold_split(X, y, k)
    fold_losses = []
    fold_mse_progress = []  # Store MSE progress for each fold

    # Create subplots for each fold
    n_cols = 2
    n_rows = (k + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for fold, (X_train, y_train, X_val, y_val) in enumerate(folds):
        print(f"\nStarting fold {fold + 1}/{k}")
        
        # Train with PSO on the current fold
        best_model, best_loss, mse_progress = pso_train(X_train, y_train, layer_sizes, num_particles, max_iter, inertia, cognitive_const, social_const)
        
        # Store MSE progress for plotting
        fold_mse_progress.append(mse_progress)
        
        # Validation loss on the current fold
        y_pred = best_model.forward(X_val)
        val_loss = best_model.loss(y_pred, y_val)
        fold_losses.append(val_loss)
        
        print(f"Fold {fold + 1} Validation MSE: {val_loss}")
        
        # Plot desired vs. predicted output for the current fold in a subplot
        axes[fold].plot(y_val, label='Desired Output', color='blue')
        axes[fold].plot(y_pred, label='Predicted Output', color='red')
        axes[fold].set_title(f'Fold {fold + 1} Desired vs. Predicted Output')
        axes[fold].set_xlabel('Sample Index')
        axes[fold].set_ylabel('Output Value')
        axes[fold].legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    avg_loss = np.mean(fold_losses)
    print(f"\nAverage Validation Loss across {k} folds: {avg_loss}")
    
    # Plot MSE progress for each fold
    plot_mse_progress(fold_mse_progress)
    
    return avg_loss

def plot_mse_progress(fold_mse_progress):
    plt.figure(figsize=(10, 6))
    for fold, mse_progress in enumerate(fold_mse_progress):
        plt.plot(mse_progress, label=f'Fold {fold + 1}')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE Progress Across Folds')
    plt.legend()
    plt.show()

# Example usage with random dataset
def read_data():
    filepath = "AirQualityUCI.xlsx"
    Col_T = [3,6,8,10,11,12,13,14]
    col_DO = [5]
    X_train = pd.read_excel(filepath, usecols=Col_T)
    Y_train = pd.read_excel(filepath, usecols=col_DO)
    X_train = X_train.values
    Y_train = Y_train.values
    X_f = []
    Y_f = []
    for i in range(len(X_train)):
        nega = 0
        for j in range(len(X_train[0])):
            if X_train[i][j]<0:
                nega+=1
        if (nega == 0 and Y_train[i] >0  ):
            X_f.append(X_train[i])
            Y_f.append(Y_train[i])

    return np.array(X_f),np.array(Y_f)

X, y = read_data()
input_size = X.shape[1]
output_size = y.shape[1]

# Example architecture with 2 hidden layers of sizes 10 and 5
layer_sizes = [input_size, 10,5, output_size]

# Run 10-fold cross-validation
avg_loss = k_fold_cross_validation(X, y, layer_sizes)
print(f"Average Validation Loss: {avg_loss}")
