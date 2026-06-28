#!/usr/bin/env python3
"""
Script to remove all files that have exactly two underscores in their names.
"""

import os
import glob
from pathlib import Path


def remove_files_with_two_underscores(directory_path='.', dry_run=True):
    """
    Remove all files that contain exactly two underscores in their names.

    Args:
        directory_path (str): Directory to search in (default: current directory)
        dry_run (bool): If True, only show what would be deleted without actually deleting

    Returns:
        list: List of files that were deleted (or would be deleted in dry run mode)
    """

    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        return []

    # Find all files in the directory
    all_files = [f for f in directory.iterdir() if f.is_file()]

    # Filter files that have exactly 2 underscores
    files_to_delete = []
    for file_path in all_files:
        filename = file_path.name
        underscore_count = filename.count('_')

        if underscore_count == 2:
            files_to_delete.append(file_path)

    print(f"Found {len(files_to_delete)} files with exactly 2 underscores in '{directory}':")

    if not files_to_delete:
        print("No files found with exactly 2 underscores.")
        return []

    deleted_files = []

    for file_path in files_to_delete:
        print(f"  - {file_path.name}")

        if not dry_run:
            try:
                file_path.unlink()  # Delete the file
                deleted_files.append(str(file_path))
                print(f"    ✓ Deleted")
            except Exception as e:
                print(f"    ✗ Error deleting: {e}")
        else:
            deleted_files.append(str(file_path))
            print(f"    [DRY RUN] Would delete")

    if dry_run:
        print(f"\nDRY RUN: {len(deleted_files)} files would be deleted.")
        print("Run with dry_run=False to actually delete the files.")
    else:
        print(f"\n✅ Successfully deleted {len(deleted_files)} files.")

    return deleted_files


def remove_files_recursive(directory_path='.', dry_run=True):
    """
    Recursively remove all files with exactly two underscores in all subdirectories.

    Args:
        directory_path (str): Root directory to search in
        dry_run (bool): If True, only show what would be deleted

    Returns:
        list: List of all files that were deleted (or would be deleted)
    """

    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        return []

    all_deleted = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        current_dir = Path(root)

        files_to_delete = []
        for filename in files:
            if filename.count('_') == 2:
                files_to_delete.append(current_dir / filename)

        if files_to_delete:
            print(f"\nIn directory: {current_dir}")
            print(f"Found {len(files_to_delete)} files with exactly 2 underscores:")

            for file_path in files_to_delete:
                print(f"  - {file_path.name}")

                if not dry_run:
                    try:
                        file_path.unlink()
                        all_deleted.append(str(file_path))
                        print(f"    ✓ Deleted")
                    except Exception as e:
                        print(f"    ✗ Error deleting: {e}")
                else:
                    all_deleted.append(str(file_path))
                    print(f"    [DRY RUN] Would delete")

    if not all_deleted:
        print("No files found with exactly 2 underscores.")
    elif dry_run:
        print(f"\nDRY RUN: {len(all_deleted)} files would be deleted across all directories.")
        print("Run with dry_run=False to actually delete the files.")
    else:
        print(f"\n✅ Successfully deleted {len(all_deleted)} files across all directories.")

    return all_deleted


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove files with exactly two underscores in their names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_files.py                           # Dry run in current directory
  python remove_files.py --execute                 # Actually delete in current directory
  python remove_files.py /path/to/folder          # Dry run in specific folder
  python remove_files.py /path/to/folder --execute # Actually delete in specific folder
  python remove_files.py --recursive              # Recursive dry run
  python remove_files.py --recursive --execute    # Recursive deletion
        """
    )

    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory to search in (default: current directory)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete files (default: dry run)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively in subdirectories')

    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("Use --execute flag to actually delete files")
    else:
        print("⚠️  EXECUTION MODE - Files will be permanently deleted!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return

    print(f"\nSearching for files with exactly 2 underscores...")
    print(f"Directory: {args.directory}")
    print(f"Recursive: {args.recursive}")
    print("-" * 50)

    if args.recursive:
        deleted_files = remove_files_recursive(args.directory, dry_run)
    else:
        deleted_files = remove_files_with_two_underscores(args.directory, dry_run)


if __name__ == "__main__":
    remove_files_with_two_underscores('/home/r56x196/Data/Mask_Data/Masks/combined', dry_run=False)