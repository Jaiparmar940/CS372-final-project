#!/usr/bin/env python3
"""
Plot learning curves from training logs.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.plotting import plot_learning_curves


def main():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training log CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for plot (default: show)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Metrics to plot (default: all)')
    parser.add_argument('--window', type=int, default=100,
                       help='Window size for moving average')
    args = parser.parse_args()
    
    plot_learning_curves(
        log_file=args.log_file,
        output_file=args.output,
        metrics=args.metrics,
        window_size=args.window,
    )


if __name__ == '__main__':
    main()

