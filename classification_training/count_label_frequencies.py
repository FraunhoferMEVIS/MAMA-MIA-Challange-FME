#!/usr/bin/env python3
"""
Command-line script to extract and count binary labels from JSON files.
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path


def process_json_files(folder_path):
    """
    Process all JSON files in the given folder and extract labels.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        
    Returns:
        Counter: Counter object with label counts
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)
    
    labels = []
    processed_files = 0
    error_files = []
    
    # Find all JSON files in the folder
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return Counter()
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract label
            if 'label' in data:
                labels.append(data['label'])
                processed_files += 1
            else:
                error_files.append(f"{json_file.name}: Missing 'label' key")
                
        except json.JSONDecodeError as e:
            error_files.append(f"{json_file.name}: Invalid JSON - {str(e)}")
        except Exception as e:
            error_files.append(f"{json_file.name}: Error reading file - {str(e)}")
    
    # Print processing summary
    print(f"\nProcessed {processed_files} files successfully")
    
    if error_files:
        print(f"\nErrors encountered in {len(error_files)} files:")
        for error in error_files:
            print(f"  - {error}")
    
    return Counter(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and count binary labels from JSON files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python label_counter.py /path/to/json/files
  python label_counter.py ./data --verbose
        """
    )
    
    parser.add_argument(
        'folder',
        help='Path to folder containing JSON files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Process the files
    label_counts = process_json_files(args.folder)
    
    if not label_counts:
        print("No labels found to count.")
        return
    
    # Display results
    print(f"\n{'='*50}")
    print("LABEL COUNT RESULTS")
    print(f"{'='*50}")
    
    total_files = sum(label_counts.values())
    print(f"Total files with labels: {total_files}")
    
    print(f"\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total_files) * 100
        print(f"  Label {label}: {count:,} files ({percentage:.1f}%)")
    
    if args.verbose:
        print(f"\nDetailed breakdown:")
        print(f"  Unique labels found: {list(sorted(label_counts.keys()))}")
        print(f"  Most common label: {label_counts.most_common(1)[0][0]} ({label_counts.most_common(1)[0][1]} occurrences)")
        print(f"  Least common label: {label_counts.most_common()[-1][0]} ({label_counts.most_common()[-1][1]} occurrences)")


if __name__ == "__main__":
    main()