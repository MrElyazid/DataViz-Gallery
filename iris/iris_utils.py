"""
Utility functions for Iris dataset visualization.
This module provides reusable functions for creating enhanced visualizations
of the Iris dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Optional, Union

# Set default style parameters
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Custom color palettes
IRIS_COLORS = {
    'Iris-setosa': '#FF5733',      # Coral red
    'Iris-versicolor': '#33A1FF',  # Sky blue
    'Iris-virginica': '#47D147'    # Green
}

def load_iris_data(filepath: str) -> pd.DataFrame:
    """
    Load the Iris dataset from a CSV file.
    
    Args:
        filepath: Path to the Iris dataset CSV file
        
    Returns:
        DataFrame containing the Iris dataset
    """
    df = pd.read_csv(filepath, delimiter=',', 
                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    return df

def split_by_class(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split the Iris dataset by class.
    
    Args:
        df: DataFrame containing the Iris dataset
        
    Returns:
        Dictionary with class names as keys and corresponding DataFrames as values
    """
    classes = df['class'].unique()
    return {cls: df[df['class'] == cls] for cls in classes}

def plot_feature_distributions(class_df: pd.DataFrame, class_name: str, color: str) -> plt.Figure:
    """
    Create a 2x2 grid of histograms showing the distribution of all features for a specific Iris class.
    
    Args:
        class_df: DataFrame containing data for a specific Iris class
        class_name: Name of the Iris class
        color: Color to use for the histograms
        
    Returns:
        Matplotlib Figure object
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    step = 0.1
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    titles = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    for i, (feature, title) in enumerate(zip(features, titles)):
        row, col = i // 2, i % 2
        
        xmin, xmax = class_df[feature].min(), class_df[feature].max()
        bins = np.arange(xmin, xmax + 2*step, step)
        
        # Create histogram with custom styling
        axs[row, col].hist(class_df[feature], bins=bins, 
                          color=color, alpha=0.7,
                          linewidth=0.25, edgecolor="white")
        
        # Add kernel density estimate
        class_df[feature].plot.kde(ax=axs[row, col], color='black', linewidth=2)
        
        # Set ticks and labels
        axs[row, col].set_xticks(np.arange(xmin, xmax + step, step))
        axs[row, col].set_title(f"{title} Distribution", fontsize=16)
        axs[row, col].set_xlabel(f"{title} (cm)", fontsize=14)
        axs[row, col].set_ylabel("Frequency", fontsize=14)
        axs[row, col].grid(True, linestyle='--', alpha=0.7)
        
        # Add mean line
        mean_val = class_df[feature].mean()
        axs[row, col].axvline(mean_val, color='red', linestyle='--', linewidth=1.5)
        axs[row, col].text(mean_val, axs[row, col].get_ylim()[1]*0.9, 
                          f'Mean: {mean_val:.2f}', 
                          color='red', fontweight='bold',
                          ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    fig.suptitle(f'Distribution of Sepal and Petal Dimensions for {class_name} Class', 
                fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def plot_feature_comparison(df: pd.DataFrame, feature_x: str, feature_y: str) -> plt.Figure:
    """
    Create a scatter plot comparing two features across all Iris classes.
    
    Args:
        df: DataFrame containing the Iris dataset
        feature_x: Feature to plot on the x-axis
        feature_y: Feature to plot on the y-axis
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Split data by class
    class_dfs = split_by_class(df)
    
    # Plot each class with a different color
    for class_name, class_df in class_dfs.items():
        ax.scatter(class_df[feature_x], class_df[feature_y], 
                  c=IRIS_COLORS[class_name], 
                  label=class_name.replace('Iris-', ''),
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add feature means for each class as larger points
    for class_name, class_df in class_dfs.items():
        mean_x = class_df[feature_x].mean()
        mean_y = class_df[feature_y].mean()
        ax.scatter(mean_x, mean_y, 
                  c='black', 
                  s=150, 
                  marker='X',
                  edgecolors=IRIS_COLORS[class_name], 
                  linewidth=2,
                  label=f"{class_name.replace('Iris-', '')} Mean")
    
    # Add grid, labels, and title
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel(f"{feature_x.replace('_', ' ').title()} (cm)", fontsize=14)
    ax.set_ylabel(f"{feature_y.replace('_', ' ').title()} (cm)", fontsize=14)
    ax.set_title(f"Comparison of {feature_x.replace('_', ' ').title()} vs {feature_y.replace('_', ' ').title()} by Iris Class", 
                fontsize=16, fontweight='bold')
    
    # Add legend with custom styling
    legend = ax.legend(frameon=True, fontsize=12, loc='best', 
                      facecolor='white', edgecolor='gray')
    
    # Add correlation coefficient annotation
    corr = df[[feature_x, feature_y]].corr().iloc[0, 1]
    ax.annotate(f'Correlation: {corr:.2f}', 
               xy=(0.05, 0.95), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
               fontsize=12, ha='left', va='top')
    
    plt.tight_layout()
    
    return fig

def create_pairplot(df: pd.DataFrame) -> plt.Figure:
    """
    Create an enhanced pairplot for the Iris dataset using Seaborn.
    
    Args:
        df: DataFrame containing the Iris dataset
        
    Returns:
        Seaborn PairGrid object
    """
    # Create a custom colormap for the diagonal
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Create the pairplot with custom styling
    g = sns.pairplot(df, 
                    hue='class', 
                    vars=features,
                    palette=IRIS_COLORS,
                    diag_kind='kde',
                    plot_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'white', 'linewidth': 0.5},
                    diag_kws={'fill': True, 'linewidth': 2, 'alpha': 0.5},
                    height=2.5)
    
    # Enhance the plot with titles and styling
    g.fig.suptitle('Pairwise Relationships Between Iris Features', 
                  fontsize=20, fontweight='bold', y=1.02)
    
    # Add correlation coefficients to the upper triangle
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            if i < j:  # Upper triangle
                ax = g.axes[i, j]
                corr = df[[feature_i, feature_j]].corr().iloc[0, 1]
                ax.annotate(f'r = {corr:.2f}', 
                           xy=(0.5, 0.9), xycoords='axes fraction',
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           fontsize=10)
    
    # Improve axis labels
    for i, feature in enumerate(features):
        for j in range(len(features)):
            if i == len(features) - 1:  # Bottom row
                g.axes[i, j].set_xlabel(feature.replace('_', ' ').title() + ' (cm)', fontsize=12)
            if j == 0:  # Leftmost column
                g.axes[i, j].set_ylabel(features[i].replace('_', ' ').title() + ' (cm)', fontsize=12)
    
    # Adjust legend
    g._legend.set_title('Iris Class')
    for t, l in zip(g._legend.texts, ['Setosa', 'Versicolor', 'Virginica']):
        t.set_text(l)
    
    plt.tight_layout()
    
    return g

def plot_3d_scatter(df: pd.DataFrame, x: str, y: str, z: str) -> plt.Figure:
    """
    Create a 3D scatter plot for three selected features.
    
    Args:
        df: DataFrame containing the Iris dataset
        x: Feature to plot on the x-axis
        y: Feature to plot on the y-axis
        z: Feature to plot on the z-axis
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Split data by class
    class_dfs = split_by_class(df)
    
    # Plot each class with a different color
    for class_name, class_df in class_dfs.items():
        ax.scatter(class_df[x], class_df[y], class_df[z],
                  c=IRIS_COLORS[class_name],
                  label=class_name.replace('Iris-', ''),
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add feature means for each class as larger points
    for class_name, class_df in class_dfs.items():
        mean_x = class_df[x].mean()
        mean_y = class_df[y].mean()
        mean_z = class_df[z].mean()
        ax.scatter(mean_x, mean_y, mean_z,
                  c='black',
                  s=150,
                  marker='X',
                  edgecolors=IRIS_COLORS[class_name],
                  linewidth=2,
                  label=f"{class_name.replace('Iris-', '')} Mean")
    
    # Add labels and title
    ax.set_xlabel(f"{x.replace('_', ' ').title()} (cm)", fontsize=12)
    ax.set_ylabel(f"{y.replace('_', ' ').title()} (cm)", fontsize=12)
    ax.set_zlabel(f"{z.replace('_', ' ').title()} (cm)", fontsize=12)
    ax.set_title(f"3D Visualization of Iris Features", fontsize=16, fontweight='bold')
    
    # Add legend
    ax.legend(frameon=True, fontsize=10, loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Improve perspective
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    return fig