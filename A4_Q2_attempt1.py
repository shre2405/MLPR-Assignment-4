import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import os

# Fix working directory issue - set to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Set random seed for reproducibility
np.random.seed(42)

# Create results directory
os.makedirs('results/q2_gmm', exist_ok=True)

print("="*70)
print(" QUESTION 2: GMM-BASED IMAGE SEGMENTATION")
print("="*70)
print(f"Working directory: {os.getcwd()}\n")

#############################################
# STEP 1: LOAD AND PREPROCESS IMAGE
#############################################

# Image filename - CHANGE THIS to try different images!
# Options: 12003A.jpg, 187039A.jpg, 20008.jpg, 277095.jpg, 46076.jpg, 68077A.jpg, 94079A.jpg
IMAGE_FILENAME = '94079A.jpg'

img_path = os.path.join('data', IMAGE_FILENAME)

print(f"[1/6] Loading image: {IMAGE_FILENAME}")
print(f"  Looking for: {os.path.abspath(img_path)}")

try:
    # Check if file exists first
    if not os.path.exists(img_path):
        print(f"\n❌ ERROR: Image not found!")
        print(f"  Expected path: {os.path.abspath(img_path)}")
        print(f"\n  Checking data folder...")
        if os.path.exists('data'):
            files = os.listdir('data')
            print(f"  Files in data folder: {files}")
        else:
            print(f"  ❌ 'data' folder doesn't exist!")
        exit(1)
    
    img = Image.open(img_path)
    img_array = np.array(img)
    
    print(f"✓ Image loaded successfully!")
    print(f"  Size: {img.size[0]} x {img.size[1]} pixels")
    print(f"  Shape: {img_array.shape}")
    print(f"  Total pixels: {img.size[0] * img.size[1]:,}")
    
except Exception as e:
    print(f"❌ ERROR loading image: {e}")
    exit(1)

# Display original image
plt.figure(figsize=(8, 6))
plt.imshow(img_array)
plt.title(f"Original Image: {IMAGE_FILENAME}")
plt.axis('off')
plt.tight_layout()
plt.savefig('results/q2_gmm/original_image.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: results/q2_gmm/original_image.png")

#############################################
# STEP 2: EXTRACT 5D FEATURES
#############################################

print("\n[2/6] Extracting 5D features (row, col, R, G, B)...")

height, width, channels = img_array.shape

# Create feature vectors for each pixel
features = []

for i in range(height):
    for j in range(width):
        row_idx = i
        col_idx = j
        r, g, b = img_array[i, j]
        
        # Create 5D feature vector
        feature = [row_idx, col_idx, r, g, b]
        features.append(feature)

features = np.array(features, dtype=float)
print(f"✓ Extracted features shape: {features.shape}")
print(f"  (Each of {features.shape[0]} pixels has 5 features)")

#############################################
# STEP 3: NORMALIZE FEATURES TO [0, 1]
#############################################

print("\n[3/6] Normalizing features to [0, 1]...")

# Normalize each dimension independently
features_normalized = features.copy()

# Normalize row indices (0 to height-1) → [0, 1]
features_normalized[:, 0] = features[:, 0] / (height - 1) if height > 1 else 0

# Normalize column indices (0 to width-1) → [0, 1]
features_normalized[:, 1] = features[:, 1] / (width - 1) if width > 1 else 0

# Normalize R, G, B (0 to 255) → [0, 1]
features_normalized[:, 2] = features[:, 2] / 255.0  # R
features_normalized[:, 3] = features[:, 3] / 255.0  # G
features_normalized[:, 4] = features[:, 4] / 255.0  # B

print(f"✓ Features normalized!")
print(f"  Feature ranges:")
print(f"    Row index: [{features_normalized[:, 0].min():.3f}, {features_normalized[:, 0].max():.3f}]")
print(f"    Col index: [{features_normalized[:, 1].min():.3f}, {features_normalized[:, 1].max():.3f}]")
print(f"    Red:       [{features_normalized[:, 2].min():.3f}, {features_normalized[:, 2].max():.3f}]")
print(f"    Green:     [{features_normalized[:, 3].min():.3f}, {features_normalized[:, 3].max():.3f}]")
print(f"    Blue:      [{features_normalized[:, 4].min():.3f}, {features_normalized[:, 4].max():.3f}]")

#############################################
# STEP 4: GMM MODEL ORDER SELECTION WITH K-FOLD CV
#############################################

print("\n[4/6] GMM Model Order Selection with 5-Fold Cross-Validation...")
print("  Objective: Maximize average validation log-likelihood")

# Define range of K (number of components) to try
K_range = [2, 3, 4, 5, 6, 7, 8]
k_folds = 5

print(f"  Testing K values: {K_range}")
print(f"  Using {k_folds}-fold cross-validation")

# Store results
cv_scores = []
cv_std = []

for K in K_range:
    print(f"\n  Testing K={K} components...")
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(features_normalized)):
        # Split data
        X_train_fold = features_normalized[train_idx]
        X_val_fold = features_normalized[val_idx]
        
        # Fit GMM on training fold
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            random_state=42,
            max_iter=100,
            n_init=3
        )
        gmm.fit(X_train_fold)
        
        # Compute log-likelihood on validation fold
        val_log_likelihood = gmm.score(X_val_fold)
        fold_scores.append(val_log_likelihood)
    
    # Average across folds
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    cv_scores.append(mean_score)
    cv_std.append(std_score)
    
    print(f"    Mean validation log-likelihood: {mean_score:.4f} (±{std_score:.4f})")

# Find best K
best_K_idx = np.argmax(cv_scores)
best_K = K_range[best_K_idx]
best_score = cv_scores[best_K_idx]

print("\n" + "-"*70)
print(" GMM CROSS-VALIDATION RESULTS")
print("-"*70)
print(f"  Best K (number of components): {best_K}")
print(f"  Best average log-likelihood: {best_score:.4f}")
print("-"*70)

#############################################
# STEP 5: VISUALIZE CV RESULTS
#############################################

print("\n[5/6] Visualizing cross-validation results...")

plt.figure(figsize=(12, 7))

# Plot with error bars
plt.errorbar(K_range, cv_scores, yerr=cv_std, 
             marker='o', markersize=8, linewidth=2, capsize=5,
             label='Validation Log-Likelihood ± Std')

# Mark best K
plt.scatter(best_K, best_score, 
            s=300, c='red', marker='*', edgecolors='black', linewidths=2,
            label=f'Best K={best_K}', zorder=5)

plt.xlabel('Number of GMM Components (K)', fontsize=12)
plt.ylabel('Average Validation Log-Likelihood', fontsize=12)
plt.title('GMM Model Order Selection via Cross-Validation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('results/q2_gmm/gmm_cv_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: results/q2_gmm/gmm_cv_results.png")

#############################################
# STEP 6: TRAIN FINAL GMM AND SEGMENT IMAGE
#############################################

print(f"\n[6/6] Training final GMM with K={best_K} on all data...")

# Fit final GMM with best K
final_gmm = GaussianMixture(
    n_components=best_K,
    covariance_type='full',
    random_state=42,
    max_iter=200,
    n_init=5
)
final_gmm.fit(features_normalized)

print(f"✓ Final GMM trained!")
print(f"  Converged: {final_gmm.converged_}")
print(f"  Iterations: {final_gmm.n_iter_}")

# Predict component labels for each pixel
print("\nAssigning segment labels to pixels...")
labels = final_gmm.predict(features_normalized)

# Reshape labels back to image dimensions
labels_image = labels.reshape(height, width)

print(f"✓ Segmentation complete!")
print(f"  Unique segments: {np.unique(labels)}")
print(f"  Segment distribution:")
for seg_id in np.unique(labels):
    count = np.sum(labels == seg_id)
    percentage = (count / len(labels)) * 100
    print(f"    Segment {seg_id}: {count} pixels ({percentage:.1f}%)")

#############################################
# STEP 7: VISUALIZE SEGMENTATION
#############################################

print("\nVisualizing segmentation results...")

# Create segmentation visualization with good contrast
# Map segment labels uniformly across grayscale range
unique_labels = np.unique(labels_image)
n_segments = len(unique_labels)

# Create colormap for segments
segment_colors = np.linspace(0, 255, n_segments, dtype=int)
labels_display = np.zeros_like(labels_image, dtype=np.uint8)

for idx, label in enumerate(unique_labels):
    labels_display[labels_image == label] = segment_colors[idx]

# Create side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original image
axes[0].imshow(img_array)
axes[0].set_title(f'Original Image\n{IMAGE_FILENAME}', fontsize=14)
axes[0].axis('off')

# Segmentation
axes[1].imshow(labels_display, cmap='nipy_spectral')
axes[1].set_title(f'GMM Segmentation (K={best_K} components)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/q2_gmm/segmentation_result.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: results/q2_gmm/segmentation_result.png")

# Also create individual segment visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original
axes[0].imshow(img_array)
axes[0].set_title('Original Image', fontsize=12)
axes[0].axis('off')

# Segmentation (color)
axes[1].imshow(labels_display, cmap='nipy_spectral')
axes[1].set_title(f'Segmentation (K={best_K})', fontsize=12)
axes[1].axis('off')

# Segmentation (grayscale)
axes[2].imshow(labels_display, cmap='gray')
axes[2].set_title(f'Segmentation (Grayscale)', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/q2_gmm/segmentation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: results/q2_gmm/segmentation_comparison.png")

#############################################
# FINAL SUMMARY
#############################################

print("\n" + "="*70)
print(" QUESTION 2 COMPLETE!")
print("="*70)
print(f"  Image: {IMAGE_FILENAME}")
print(f"  Image size: {width}x{height} pixels")
print(f"  Optimal K: {best_K} components")
print(f"  Best log-likelihood: {best_score:.4f}")
print(f"  Number of segments: {n_segments}")
print("="*70)
print("\n✓✓✓ All results saved in 'results/q2_gmm/' ✓✓✓")
print("\nGenerated files:")
print("  1. original_image.png - Your input image")
print("  2. gmm_cv_results.png - Model order selection plot")
print("  3. segmentation_result.png - Original vs Segmentation")
print("  4. segmentation_comparison.png - Three-way comparison")
print("="*70)