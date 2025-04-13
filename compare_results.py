import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_results(result_paths):
    """
    Load model results from one or more JSON files.
    
    Args:
        result_paths (list): List of paths to result JSON files
        
    Returns:
        dict: Combined results from all files
    """
    all_results = {}
    
    for path in result_paths:
        with open(path, 'r') as f:
            results = json.load(f)
            
        # Extract experiment name from filename
        filename = os.path.basename(path)
        experiment_name = filename.split('_comparison_')[0] if '_comparison_' in filename else filename.split('.')[0]
        
        # Add experiment prefix to model names to avoid conflicts
        prefixed_results = {}
        for model_name, model_results in results.items():
            prefixed_name = f"{experiment_name}_{model_name}"
            prefixed_results[prefixed_name] = model_results
            
        all_results.update(prefixed_results)
    
    return all_results

def create_comparison_dataframe(results):
    """
    Create a DataFrame for model comparison from results.
    
    Args:
        results (dict): Model results dictionary
        
    Returns:
        pandas.DataFrame: Comparison DataFrame
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        
        # Extract experiment name and model type
        parts = model_name.split('_')
        if len(parts) >= 2:
            experiment = parts[0]
            model_type = '_'.join(parts[1:])
        else:
            experiment = 'default'
            model_type = model_name
            
        row = {
            'experiment': experiment,
            'model': model_type,
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'cv_r2_mean': metrics['cv_r2_mean'],
            'cv_r2_std': metrics['cv_r2_std'],
            'train_time': metrics.get('train_time', 0)
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def plot_metric_comparison(df, metric='r2', ascending=False, title=None):
    """
    Plot a comparison of models based on a specific metric.
    
    Args:
        df (pandas.DataFrame): Comparison DataFrame
        metric (str): Metric to compare
        ascending (bool): Sort order
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    # Sort by the specified metric
    df_sorted = df.sort_values(by=metric, ascending=ascending)
    
    # Create a categorical color palette based on unique experiments
    unique_experiments = df_sorted['experiment'].unique()
    palette = sns.color_palette("husl", len(unique_experiments))
    experiment_colors = dict(zip(unique_experiments, palette))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create barplot
    bars = ax.barh(
        df_sorted['model'], 
        df_sorted[metric],
        color=[experiment_colors[exp] for exp in df_sorted['experiment']]
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center')
    
    # Add a legend for experiments
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in experiment_colors.values()]
    ax.legend(legend_handles, experiment_colors.keys(), title='Feature Set')
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Model Comparison by {metric.upper()}", fontsize=14)
    
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_metrics_grid(df, metrics=['r2', 'rmse', 'mae', 'train_time'], title=None):
    """
    Plot a grid of comparisons for multiple metrics.
    
    Args:
        df (pandas.DataFrame): Comparison DataFrame
        metrics (list): List of metrics to compare
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: Grid plot
    """
    # Determine grid dimensions
    n_metrics = len(metrics)
    n_rows = (n_metrics + 1) // 2  # Ceiling division
    n_cols = min(2, n_metrics)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    axes = axes.flatten()
    
    # Create a categorical color palette based on unique experiments
    unique_experiments = df['experiment'].unique()
    palette = sns.color_palette("husl", len(unique_experiments))
    experiment_colors = dict(zip(unique_experiments, palette))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Sort by metric (ascending for error metrics, descending for R²)
            ascending = metric not in ['r2', 'cv_r2_mean']
            df_sorted = df.sort_values(by=metric, ascending=ascending)
            
            # Create barplot
            bars = ax.barh(
                df_sorted['model'], 
                df_sorted[metric],
                color=[experiment_colors[exp] for exp in df_sorted['experiment']]
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                        va='center', fontsize=9)
            
            # Set title and labels
            ax.set_title(f"Comparison by {metric.upper()}", fontsize=12)
            ax.set_xlabel(metric.upper(), fontsize=10)
            
            # Only show y-axis labels on the first column
            if i % n_cols == 0:
                ax.set_ylabel('Model', fontsize=10)
            else:
                ax.set_ylabel('')
                
    # Add a single legend for the entire figure
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in experiment_colors.values()]
    fig.legend(legend_handles, experiment_colors.keys(), title='Feature Set', 
               loc='lower center', ncol=len(experiment_colors), bbox_to_anchor=(0.5, 0))
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Compare model evaluation results')
    
    # Result files to compare
    parser.add_argument('--files', nargs='+', 
                        help='Specific result JSON files to compare')
    
    # Or use all results in a directory
    parser.add_argument('--dir', default='output/model_results',
                        help='Directory containing result files')
    
    # Metrics to compare
    parser.add_argument('--metrics', nargs='+', default=['r2', 'rmse', 'mae', 'train_time'],
                        help='Metrics to compare')
    
    # Output file
    parser.add_argument('--output', 
                        help='Path to save comparison plot')
    
    args = parser.parse_args()
    
    # Get result files
    if args.files:
        result_files = args.files
    else:
        result_files = glob(os.path.join(args.dir, '*comparison*.json'))
        
    if not result_files:
        print(f"No result files found in {args.dir}")
        return
        
    print(f"Comparing {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load results
    results = load_results(result_files)
    
    # Create comparison DataFrame
    df_comparison = create_comparison_dataframe(results)
    
    print("\nComparison summary:")
    print(df_comparison[['experiment', 'model', 'r2', 'rmse', 'mae']].sort_values('r2', ascending=False))
    
    # Plot metrics grid
    fig = plot_metrics_grid(df_comparison, metrics=args.metrics)
    
    # Save or show plot
    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()
    
if __name__ == '__main__':
    main()