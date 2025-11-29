import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create results directories
os.makedirs('results/q1_svm', exist_ok=True)
os.makedirs('results/q1_mlp', exist_ok=True)

print("="*70)
print(" QUESTION 1: SVM AND MLP CLASSIFICATION")
print("="*70)

#############################################
# STEP 1: DATA GENERATION FUNCTION
#############################################

def generate_data(n_samples, r_inner=2, r_outer=4, sigma=1):
    """
    Generate concentric circle data with noise
    
    Parameters:
    - n_samples: total number of samples
    - r_inner: radius for class -1 (inner circle)
    - r_outer: radius for class +1 (outer circle)
    - sigma: standard deviation of Gaussian noise
    
    Returns:
    - X: feature matrix (n_samples, 2)
    - y: labels (n_samples,) with values {-1, +1}
    """
    
    # Half samples for each class
    n_per_class = n_samples // 2
    
    # Generate class -1 (inner circle)
    theta_inner = np.random.uniform(-np.pi, np.pi, n_per_class)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)
    
    # Add Gaussian noise
    noise_inner = np.random.normal(0, sigma, (n_per_class, 2))
    X_inner = np.column_stack([x_inner, y_inner]) + noise_inner
    y_inner = -np.ones(n_per_class)
    
    # Generate class +1 (outer circle)
    theta_outer = np.random.uniform(-np.pi, np.pi, n_per_class)
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)
    
    # Add Gaussian noise
    noise_outer = np.random.normal(0, sigma, (n_per_class, 2))
    X_outer = np.column_stack([x_outer, y_outer]) + noise_outer
    y_outer = np.ones(n_per_class)
    
    # Combine both classes
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

#############################################
# STEP 2: GENERATE TRAIN AND TEST DATA
#############################################

print("\n[1/8] Generating data...")
X_train, y_train = generate_data(n_samples=1000, r_inner=2, r_outer=4, sigma=1)
X_test, y_test = generate_data(n_samples=10000, r_inner=2, r_outer=4, sigma=1)

print(f"✓ Training data: {X_train.shape}, Labels: {y_train.shape}")
print(f"✓ Test data: {X_test.shape}, Labels: {y_test.shape}")
print(f"✓ Training class distribution: {np.sum(y_train == -1)} inner, {np.sum(y_train == 1)} outer")

#############################################
# STEP 3: VISUALIZE THE DATA
#############################################

def plot_data(X, y, title="Data Distribution", filename=None):
    """Plot the 2D data with different colors for each class"""
    plt.figure(figsize=(10, 8))
    
    # Separate classes
    X_inner = X[y == -1]
    X_outer = X[y == 1]
    
    # Plot
    plt.scatter(X_inner[:, 0], X_inner[:, 1], c='blue', alpha=0.5, 
                label='Class -1 (Inner)', s=20)
    plt.scatter(X_outer[:, 0], X_outer[:, 1], c='red', alpha=0.5, 
                label='Class +1 (Outer)', s=20)
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close()

print("\n[2/8] Visualizing data...")
plot_data(X_train, y_train, 
          title="Training Data (1000 samples)", 
          filename="results/q1_svm/training_data.png")

#############################################
# PART A: SVM WITH GAUSSIAN (RBF) KERNEL
#############################################

print("\n" + "="*70)
print(" PART A: SVM WITH RBF KERNEL")
print("="*70)

#############################################
# STEP 4: SVM HYPERPARAMETER GRID SEARCH WITH K-FOLD CV
#############################################

# Define hyperparameter grid
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.01, 0.1, 0.5, 1, 2, 5, 10]
}

print("\n[3/8] SVM Hyperparameter Grid Search with 5-Fold CV...")
print(f"  → C values: {param_grid_svm['C']}")
print(f"  → Gamma values: {param_grid_svm['gamma']}")
print(f"  → Total combinations: {len(param_grid_svm['C']) * len(param_grid_svm['gamma'])}")

# K-Fold Cross-Validation setup
k_folds = 5

# GridSearchCV with K-fold CV
svm_grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf'),
    param_grid=param_grid_svm,
    cv=k_folds,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("  → Training (this may take 30-60 seconds)...")
svm_grid_search.fit(X_train, y_train)

# Best parameters
best_svm_params = svm_grid_search.best_params_
best_svm_score = svm_grid_search.best_score_

print("\n" + "-"*70)
print(" SVM CROSS-VALIDATION RESULTS")
print("-"*70)
print(f"  Best C (Box Constraint): {best_svm_params['C']}")
print(f"  Best Gamma (Kernel Width): {best_svm_params['gamma']}")
print(f"  Best CV Accuracy: {best_svm_score:.4f}")
print(f"  Best CV Error: {1 - best_svm_score:.4f}")
print("-"*70)

#############################################
# STEP 5: VISUALIZE SVM CROSS-VALIDATION RESULTS
#############################################

def plot_cv_results_svm(grid_search, filename=None):
    """Plot heatmap of cross-validation results"""
    results = grid_search.cv_results_
    
    # Extract mean test scores
    scores = results['mean_test_score'].reshape(
        len(param_grid_svm['C']), 
        len(param_grid_svm['gamma'])
    )
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    im = plt.imshow(scores, interpolation='nearest', cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Cross-Validation Accuracy')
    
    # Set ticks and labels
    plt.xticks(range(len(param_grid_svm['gamma'])), param_grid_svm['gamma'])
    plt.yticks(range(len(param_grid_svm['C'])), param_grid_svm['C'])
    plt.xlabel('Gamma (Kernel Width)', fontsize=12)
    plt.ylabel('C (Box Constraint)', fontsize=12)
    plt.title('SVM Cross-Validation Accuracy Heatmap', fontsize=14)
    
    # Annotate cells with scores
    for i in range(len(param_grid_svm['C'])):
        for j in range(len(param_grid_svm['gamma'])):
            text = plt.text(j, i, f'{scores[i, j]:.3f}',
                          ha="center", va="center", color="white", fontsize=9)
    
    # Mark best parameters
    best_idx = np.unravel_index(scores.argmax(), scores.shape)
    plt.scatter(best_idx[1], best_idx[0], marker='*', s=500, 
                c='red', edgecolors='white', linewidths=2,
                label=f'Best: C={best_svm_params["C"]}, γ={best_svm_params["gamma"]}')
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close()

print("\n[4/8] Visualizing SVM cross-validation results...")
plot_cv_results_svm(svm_grid_search, 
                    filename='results/q1_svm/svm_cv_heatmap.png')

#############################################
# STEP 6: TRAIN FINAL SVM MODEL WITH BEST PARAMETERS
#############################################

print("\n[5/8] Training final SVM model with best parameters...")
best_svm_model = SVC(kernel='rbf', 
                     C=best_svm_params['C'], 
                     gamma=best_svm_params['gamma'])
best_svm_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_svm = best_svm_model.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
test_error_svm = 1 - test_accuracy_svm

print("\n" + "="*70)
print(" SVM TEST SET PERFORMANCE")
print("="*70)
print(f"  Test Accuracy: {test_accuracy_svm:.4f} ({test_accuracy_svm*100:.2f}%)")
print(f"  Test Error (Probability of Error): {test_error_svm:.4f} ({test_error_svm*100:.2f}%)")
print(f"  Number of support vectors: {len(best_svm_model.support_)}")
print("="*70)

#############################################
# STEP 7: VISUALIZE SVM DECISION BOUNDARY
#############################################

def plot_decision_boundary(model, X, y, title, filename=None, sample_size=2000):
    """Plot decision boundary with data points"""
    
    # Sample data for visualization if too large
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
    else:
        X_plot = X
        y_plot = y
    
    plt.figure(figsize=(12, 10))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05  # mesh step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=[-1.5, 0, 1.5])
    plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0])
    
    # Plot data points
    X_inner = X_plot[y_plot == -1]
    X_outer = X_plot[y_plot == 1]
    
    plt.scatter(X_inner[:, 0], X_inner[:, 1], c='blue', alpha=0.6,
                label='Class -1 (Inner)', s=20, edgecolors='k', linewidths=0.5)
    plt.scatter(X_outer[:, 0], X_outer[:, 1], c='red', alpha=0.6,
                label='Class +1 (Outer)', s=20, edgecolors='k', linewidths=0.5)
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close()

print("\n[6/8] Visualizing SVM decision boundary...")
plot_decision_boundary(
    best_svm_model, 
    X_test, 
    y_test,
    title=f'SVM Decision Boundary on Test Set\n(Accuracy: {test_accuracy_svm:.4f}, Error: {test_error_svm:.4f})',
    filename='results/q1_svm/svm_decision_boundary.png',
    sample_size=2000
)

print("\n✓ SVM training and evaluation complete!")

#############################################
# PART B: MLP (MULTI-LAYER PERCEPTRON)
#############################################

print("\n" + "="*70)
print(" PART B: MLP (MULTI-LAYER PERCEPTRON)")
print("="*70)

#############################################
# STEP 8: MLP HYPERPARAMETER SEARCH WITH K-FOLD CV
#############################################

# Define hyperparameter grid for MLP
# We're tuning the number of neurons in the hidden layer
param_grid_mlp = {
    'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (150,), (200,)]
}

print("\n[7/8] MLP Hyperparameter Grid Search with 5-Fold CV...")
print(f"  → Hidden layer sizes to test: {[h[0] for h in param_grid_mlp['hidden_layer_sizes']]}")
print(f"  → Total configurations: {len(param_grid_mlp['hidden_layer_sizes'])}")

# GridSearchCV for MLP
mlp_grid_search = GridSearchCV(
    estimator=MLPClassifier(
        activation='tanh',  # Good for this problem
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    ),
    param_grid=param_grid_mlp,
    cv=k_folds,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("  → Training (this may take 1-2 minutes)...")
mlp_grid_search.fit(X_train, y_train)

# Best parameters
best_mlp_params = mlp_grid_search.best_params_
best_mlp_score = mlp_grid_search.best_score_

print("\n" + "-"*70)
print(" MLP CROSS-VALIDATION RESULTS")
print("-"*70)
print(f"  Best Hidden Layer Size: {best_mlp_params['hidden_layer_sizes'][0]} neurons")
print(f"  Best CV Accuracy: {best_mlp_score:.4f}")
print(f"  Best CV Error: {1 - best_mlp_score:.4f}")
print("-"*70)

#############################################
# STEP 9: VISUALIZE MLP CROSS-VALIDATION RESULTS
#############################################

def plot_cv_results_mlp(grid_search, filename=None):
    """Plot CV results for different hidden layer sizes"""
    results = grid_search.cv_results_
    hidden_sizes = [h[0] for h in param_grid_mlp['hidden_layer_sizes']]
    mean_scores = results['mean_test_score']
    std_scores = results['std_test_score']
    
    plt.figure(figsize=(12, 7))
    
    # Plot mean accuracy with error bars
    plt.errorbar(hidden_sizes, mean_scores, yerr=std_scores, 
                 marker='o', markersize=8, linewidth=2, capsize=5,
                 label='CV Accuracy ± Std')
    
    # Mark best configuration
    best_idx = np.argmax(mean_scores)
    plt.scatter(hidden_sizes[best_idx], mean_scores[best_idx], 
                s=300, c='red', marker='*', edgecolors='black', linewidths=2,
                label=f'Best: {hidden_sizes[best_idx]} neurons', zorder=5)
    
    plt.xlabel('Number of Hidden Neurons', fontsize=12)
    plt.ylabel('Cross-Validation Accuracy', fontsize=12)
    plt.title('MLP Cross-Validation: Hidden Layer Size vs Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close()

plot_cv_results_mlp(mlp_grid_search,
                    filename='results/q1_mlp/mlp_cv_results.png')

#############################################
# STEP 10: TRAIN FINAL MLP MODEL
#############################################

print("\n[8/8] Training final MLP model with best parameters...")
best_mlp_model = MLPClassifier(
    hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'],
    activation='tanh',
    solver='adam',
    max_iter=1000,
    random_state=42
)
best_mlp_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_mlp = best_mlp_model.predict(X_test)
test_accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
test_error_mlp = 1 - test_accuracy_mlp

print("\n" + "="*70)
print(" MLP TEST SET PERFORMANCE")
print("="*70)
print(f"  Test Accuracy: {test_accuracy_mlp:.4f} ({test_accuracy_mlp*100:.2f}%)")
print(f"  Test Error (Probability of Error): {test_error_mlp:.4f} ({test_error_mlp*100:.2f}%)")
print(f"  Number of iterations: {best_mlp_model.n_iter_}")
print("="*70)

#############################################
# STEP 11: VISUALIZE MLP DECISION BOUNDARY
#############################################

print("\nVisualizing MLP decision boundary...")
plot_decision_boundary(
    best_mlp_model,
    X_test,
    y_test,
    title=f'MLP Decision Boundary on Test Set\n(Accuracy: {test_accuracy_mlp:.4f}, Error: {test_error_mlp:.4f})',
    filename='results/q1_mlp/mlp_decision_boundary.png',
    sample_size=2000
)

print("\n✓ MLP training and evaluation complete!")

#############################################
# FINAL COMPARISON
#############################################

print("\n" + "="*70)
print(" FINAL COMPARISON: SVM vs MLP")
print("="*70)
print(f"  SVM Test Error: {test_error_svm:.4f} ({test_error_svm*100:.2f}%)")
print(f"  MLP Test Error: {test_error_mlp:.4f} ({test_error_mlp*100:.2f}%)")
print(f"  Difference: {abs(test_error_svm - test_error_mlp):.4f}")
print("="*70)

print("\n✓✓✓ QUESTION 1 COMPLETE! ✓✓✓")
print("Check the 'results/' folder for all plots and visualizations.")
print("="*70)