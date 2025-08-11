import os
import argparse
import sys
import torch

# Add 'data_loader' directory to sys.path to allow direct import of its modules
# This assumes 'data_loader' is a sibling directory or already in PYTHONPATH.
# If 'data_loader' is a sub-directory of the current script's location, this might not be needed.
# If 'data_loader' contains the modules directly, use:
# from Full_loader import train_full_data_loader, test_full_data_loader
# from Clip_Loader import train_clip_data_loader, test_clip_data_loader
# The original sys.path.append was for a different structure, so I'm adapting based on typical usage.
sys.path.append(os.path.abspath('data_loader'))  # Retained original sys.path manipulation

# Import specific data loader functions from your custom modules
from data_loader.Full_loader import train_full_data_loader, test_full_data_loader
from data_loader.Clip_Loader import train_clip_data_loader, test_clip_data_loader

# Import utility functions (assuming 'utils.py' is in a location accessible via sys.path or PYTHONPATH)
from utils import seed_worker  # For reproducible DataLoader workers


# ==============================================================================
# Data Loader Function
# ==============================================================================

def get_data_loaders(args, fold_index):
    """
    Loads training and validation datasets based on specified arguments.
    Supports binary and multi-class classification tasks.
    Supports 'Clip' and 'Full' data processing modes.
    Supports datasets: "Combine", "ChangZhou", "NanJing", "SuZhou".

    Args:
        args (argparse.Namespace): Command-line arguments or configuration parameters.
        fold_index (int): The fold index for cross-validation (1-indexed).

    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoader instances for training and validation.
    """
    supported_datasets = ["Combine", "ChangZhou", "NanJing", "SuZhou"]
    if args.dataset not in supported_datasets:
        raise ValueError(f"Unsupported dataset type: {args.dataset}. Supported are: {supported_datasets}")

    supported_data_loaders = ["Clip", "Full"]
    if args.data_loader not in supported_data_loaders:
        raise ValueError(f"Unsupported data_loader type: {args.data_loader}. Supported are: {supported_data_loaders}")

    # Construct base path for dataset files
    # Example base path: /public/home/feng_rui/code/MCI_project/Dataset/Combine/Clip
    base_dataset_path = os.path.join("/public/home/feng_rui/code/MCI_project/Dataset", args.dataset, args.data_loader)

    # Determine annotation file paths based on number of classes
    # The original code had identical paths for binary and multi-class if num_classes > 2 used 'Cross_Validation_Splits2'
    # Here, simplifying to use the same 'Cross_Validation_Splits' folder as the conditions were effectively the same.
    # If a distinct 'Cross_Validation_Splits2' is intended for multi-class > 2, that logic should be reinstated.
    annotation_subdir = "Cross_Validation_Splits"  # Default subdir for annotation files

    # If num_classes > 2 specifically required a different subdir, that logic would go here.
    # e.g., if args.num_classes > 2: annotation_subdir = "Cross_Validation_Splits_MultiClass"

    train_annotation_file = os.path.join(base_dataset_path, annotation_subdir, f"MCI_set_{fold_index}_train.txt")
    test_annotation_file = os.path.join(base_dataset_path, annotation_subdir, f"MCI_set_{fold_index}_test.txt")

    num_classes_for_loader = args.num_classes  # Number of classes for the data loader
    use_augmentation = args.use_augmentation  # Whether to use data augmentation

    print(f"\n*********** Dataset: {args.dataset} ({args.data_loader}) | Fold: {fold_index} ***********")
    print(f"  Training annotation file: {train_annotation_file}")
    print(f"  Testing annotation file: {test_annotation_file}")
    print(f"  Number of classes: {num_classes_for_loader}")
    print(f"  Data augmentation: {use_augmentation}")
    print("******************************************************************\n")

    # Load data based on the chosen data loader type ('Clip' or 'Full')
    if args.data_loader == 'Clip':
        train_dataset = train_clip_data_loader(
            list_file=train_annotation_file,
            image_size=args.img_size,
            num_class=num_classes_for_loader,  # Pass the determined number of classes
            augmentation=use_augmentation
        )
        test_dataset = test_clip_data_loader(
            list_file=test_annotation_file,
            image_size=args.img_size,
            num_class=num_classes_for_loader
        )
    elif args.data_loader == 'Full':
        train_dataset = train_full_data_loader(
            list_file=train_annotation_file,
            num_segments=args.frame_num,
            duration=1,  # This seems fixed, consider making it an arg if variable
            image_size=args.img_size,
            num_class=num_classes_for_loader,
            augmentation=use_augmentation
        )
        test_dataset = test_full_data_loader(
            list_file=test_annotation_file,
            num_segments=args.frame_num,
            duration=1,
            image_size=args.img_size,
            num_class=num_classes_for_loader
        )
    else:
        # This case should be caught by the earlier check, but good for robustness
        raise ValueError(f"Invalid data_loader type: {args.data_loader}")

    # Create PyTorch DataLoader for training set
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle training data
        drop_last=args.drop_last,  # Whether to drop the last incomplete batch
        num_workers=args.workers,
        pin_memory=True,  # For faster data transfer to GPU
        worker_init_fn=seed_worker  # For reproducible results with multiple workers
    )

    # Create PyTorch DataLoader for validation/test set
    val_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle validation/test data
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    return train_loader, val_loader


# ==============================================================================
# Main Execution Block (for testing this script)
# ==============================================================================

if __name__ == "__main__":
    def parse_test_args():
        """Parses arguments for testing the get_data_loaders function."""
        parser = argparse.ArgumentParser(description="Test script for data loaders.")
        parser.add_argument('--gpu', type=str, default="0", help='GPU IDs to use (for environment setting).')
        parser.add_argument('--model', type=str, default='SlowFast', choices=['C3D', 'SlowFast', 'VideoMAE'],
                            help='Model name (not directly used by data loader).')
        parser.add_argument('--dataset', type=str, default='SuZhou',
                            choices=['Combine', 'ChangZhou', 'NanJing', 'SuZhou'], help="Dataset identifier.")
        parser.add_argument('--data_loader', type=str, default='Full', choices=['Clip', 'Full'],
                            help='Data loading mode: Clip or Full.')
        parser.add_argument('--use_augmentation', type=bool, default=True,
                            help='Use data augmentation during training.')
        # parser.add_argument('--criterion', type=str, default='BCE', choices=['BCE', 'Focal'], help='Loss criterion (not used by data loader).') # Not directly used here
        parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
        # parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (not used by data loader).')
        parser.add_argument('--batch-size', type=int, default=2,
                            help='Mini-batch size for testing.')  # Smaller batch size for quick test
        # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (not used by data loader).')
        # parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (not used by data loader).')
        # parser.add_argument('--print-freq', type=int, default=5, help='Print frequency (not used by data loader).')
        parser.add_argument('--frame_num', type=int, default=32,
                            help='Number of frames per video clip (for Full loader).')
        parser.add_argument('--img-size', type=int, default=224, help='Input image size (height and width).')
        parser.add_argument('--drop_last', type=bool, default=False,
                            help='Drop last incomplete batch.')  # Added drop_last

        # Add num_classes, as it's used by get_data_loaders
        parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for the task.')

        parsed_args = parser.parse_args()
        # Set visible GPUs using environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.gpu
        return parsed_args


    # Parse arguments for testing
    test_args = parse_test_args()

    # Example: Test with multi-class setting
    # test_args.num_classes = 4 # Override for multi-class test

    current_fold_index = 1  # Example fold index (1-indexed)

    print("--- Testing get_data_loaders ---")
    try:
        train_loader_test, val_loader_test = get_data_loaders(test_args, current_fold_index)

        print(f"\nSuccessfully created data loaders for fold {current_fold_index}.")
        print(f"  Number of training batches: {len(train_loader_test)}")
        print(f"  Number of validation batches: {len(val_loader_test)}")

        # Optionally, iterate through a few batches to check data shapes
        print("\n--- Checking a few batches ---")
        for i, (video_batch, audio_batch, label_batch) in enumerate(train_loader_test):
            print(f"Train Batch {i + 1}:")
            print(f"  Video shape: {video_batch.shape}")
            print(f"  Audio shape: {audio_batch.shape}")  # Assuming audio is returned, adjust if not
            print(f"  Label shape: {label_batch.shape}")
            if i >= 1:  # Check first 2 batches
                break

        for i, (video_batch, audio_batch, label_batch) in enumerate(val_loader_test):
            print(f"Validation Batch {i + 1}:")
            print(f"  Video shape: {video_batch.shape}")
            print(f"  Audio shape: {audio_batch.shape}")
            print(f"  Label shape: {label_batch.shape}")
            if i >= 1:  # Check first 2 batches
                break

        print("\n--- Data loader test completed successfully. ---")

    except Exception as e:
        print(f"Error during data loader test: {e}")
        import traceback

        traceback.print_exc()