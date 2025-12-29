import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA

# --- DATA UTILITIES ---

def load_split(split_name="test", base_path="embs/"):
    """Loads embeddings (float16) and labels for a specific split."""
    embs = np.load(f"{base_path}{split_name}_embs.npy").astype(np.float16)
    labs = np.load(f"{base_path}{split_name}_y.npy")
    labs = np.array([y.item() for y in labs])
    return embs, labs

def get_indices(labels):
    return {
        "pos": np.where(labels == 1)[0],
        "neg": np.where(labels == 2)[0],
        "unlabeled": np.where(labels == 3)[0]
    }

# --- TRANSFORMATION & VARIANCE ---

def run_pca(X, n_components=5):
    """Runs PCA and returns reduced data + the model for variance analysis."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    var_ratio = pca.explained_variance_ratio_
    print(f"\n[PCA Info] Components: {n_components}")
    print(f"Explained Variance (first 2): {sum(var_ratio[:2]):.2%}")
    for i, var in enumerate(var_ratio):
        print(f" - Component {i+1}: {var:.2%}")
        
    return X_pca, pca

# --- VISUALIZATION ---

def plot_component_distributions(X_reduced, indices, n_dims=3):
    """
    Plots the 1D PDF (Density) for the first N components.
    This helps identify WHICH component separates classes the best.
    """
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 4 * n_dims))
    if n_dims == 1: axes = [axes]
    
    colors = {"pos": "red", "neg": "green", "unlabeled": "gray"}
    labels = {"pos": "Ilicit", "neg": "Licit", "unlabeled": "Unlabeled"}

    for i in range(n_dims):
        for key, idx in indices.items():
            if len(idx) > 0:
                sns.kdeplot(X_reduced[idx, i], ax=axes[i], 
                            fill=True, color=colors[key], 
                            label=labels[key], alpha=0.4)
        
        axes[i].set_title(f"Distribution of Component {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_scree(pca_model):
    """Visualizes how much information is kept as we add dimensions."""
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_), 'o-', color='purple')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot: Captured Information')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- MAIN EXECUTION ---

def full_analysis_pipeline(split="test", n_dims=5):
    # 1. Load
    X_raw, y = load_split(split)
    idx = get_indices(y)
    
    # 2. PCA & Variance Analysis
    X_pca, pca_model = run_pca(X_raw, n_components=n_dims)
    
    # 3. Visualization
    # Check if more dimensions help separation
    plot_component_distributions(X_pca, idx, n_dims=min(n_dims, 4))
    
    # See how much total variance we are capturing
    plot_scree(pca_model)

# Execute
full_analysis_pipeline("test", n_dims=5)
