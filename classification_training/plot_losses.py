import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_csv_files(folder_path):
    """
    Read the loss_log.csv and loss_log_detailed.csv files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing the CSV files
        
    Returns:
        tuple: (loss_df, detailed_df) - DataFrames containing the loss data
    """
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Define file paths
    loss_file = folder / "loss_log.csv"
    detailed_file = folder / "loss_log_detailed.csv"
    
    # Check if files exist
    if not loss_file.exists():
        raise FileNotFoundError(f"loss_log.csv not found in {folder_path}")
    if not detailed_file.exists():
        raise FileNotFoundError(f"loss_log_detailed.csv not found in {folder_path}")
    
    # Read CSV files
    try:
        loss_df = pd.read_csv(loss_file)
        detailed_df = pd.read_csv(detailed_file)
    except Exception as e:
        raise Exception(f"Error reading CSV files: {e}")
    
    # Validate required columns
    if not all(col in loss_df.columns for col in ['epoch', 'train_loss', 'val_loss']):
        raise ValueError("loss_log.csv must contain columns: epoch, train_loss, val_loss")
    
    if not all(col in detailed_df.columns for col in ['epoch', 'ranking_score', 'balanced_accuracy', 'fairness_score']):
        raise ValueError("loss_log_detailed.csv must contain columns: epoch, ranking_score, balanced_accuracy, fairness_score")
    
    return loss_df, detailed_df


def create_plot(loss_df, detailed_df, output_path=None):
    """
    Create a single plot with training loss, validation loss, ranking score, balanced accuracy, and fairness score.
    
    Args:
        loss_df (pd.DataFrame): DataFrame with training and validation losses
        detailed_df (pd.DataFrame): DataFrame with detailed metrics including ranking score, balanced accuracy, and fairness score
        output_path (str, optional): Path to save the plot. If None, display the plot.
    """
    # Create figure with single plot and dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot training and validation losses on primary y-axis
    line1 = ax1.plot(loss_df['epoch'], loss_df['train_loss'], 
                     label='Training Loss', color='blue', linewidth=2)
    line2 = ax1.plot(loss_df['epoch'], loss_df['val_loss'], 
                     label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlim(1, loss_df['epoch'].max())
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for metrics (0-1 range)
    ax2 = ax1.twinx()
    line3 = ax2.plot(detailed_df['epoch'], detailed_df['ranking_score'], 
                     label='Ranking Score', color='green', linewidth=2)
    line4 = ax2.plot(detailed_df['epoch'], detailed_df['balanced_accuracy'], 
                     label='Balanced Accuracy', color='orange', linewidth=2)
    line5 = ax2.plot(detailed_df['epoch'], detailed_df['fairness_score'], 
                     label='Fairness Score', color='purple', linewidth=2)
    
    ax2.set_ylabel('Scores / Balanced Accuracy', fontsize=12, color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylim(0, 1)  # Set limits for better visibility of score metrics
    
    # Combine legends from both axes
    lines = line1 + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main function to handle command line arguments and execute the plotting."""
    parser = argparse.ArgumentParser(
        description="Plot training loss, validation loss, and ranking score from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_losses.py ./results
    python plot_losses.py /path/to/experiment/folder
    python plot_losses.py ./data --output plot.png
        """
    )
    
    parser.add_argument('input_folder', 
                        help='Path to folder containing loss_log.csv and loss_log_detailed.csv')
    
    parser.add_argument('-o', '--output',
                        type=str,
                        default='loss_plot',
                        help='Output file path for saving the plot (optional)')
    
    parser.add_argument('--format', choices=['png', 'pdf', 'svg', 'jpg'], 
                        default='png', help='Output format (default: png)')
    
    args = parser.parse_args()
    
    try:
        # Read the CSV files
        print(f"Reading CSV files from: {args.input_folder}")
        loss_df, detailed_df = read_csv_files(args.input_folder)
        
        print(f"Successfully loaded data:")
        print(f"  - loss_log.csv: {len(loss_df)} epochs")
        print(f"  - loss_log_detailed.csv: {len(detailed_df)} epochs")
        
        # Merge dataframes on epoch to ensure alignment
        merged_df = pd.merge(loss_df, detailed_df[['epoch', 'ranking_score', 'balanced_accuracy', 'fairness_score']], 
                            on='epoch', how='inner')
        
        if len(merged_df) == 0:
            raise ValueError("No matching epochs found between the two CSV files")
        
        # Prepare output path
        output_path = None
        if args.output:
            if not args.output.endswith(f'.{args.format}'):
                output_path = os.path.join(args.input_folder, f"{args.output}.{args.format}")
            else:
                output_path = os.path.join(args.input_folder, args.output)
        
        # Create and display/save the plot
        print("Creating plot...")
        create_plot(loss_df, detailed_df, output_path)
        
        if not output_path:
            print("Displaying plot...")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()