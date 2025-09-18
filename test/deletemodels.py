#!/usr/bin/env python3
"""
Delete Models Script
Removes trained models and processed data to force fresh model generation.
Run from the test directory: python deletemodels.py
"""

import sys
import shutil
from pathlib import Path

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from config import config
except ImportError as e:
    print(f"Error importing config: {e}")
    print("Make sure you're running this from the test directory and src/config is available.")
    sys.exit(1)

def main():
    """Delete models and processed data directories."""
    print("ML Model Reset Script")
    print("=" * 40)
    
    try:
        # Get paths from config
        models_dir = Path(config.get('paths', 'models', 'save_directory'))
        processed_dir = Path(config.get('paths', 'data', 'processed_directory'))
        
        # Make paths relative to project root (one level up from test/)
        project_root = Path(__file__).parent.parent
        models_path = project_root / models_dir
        processed_path = project_root / processed_dir
        
        # Also check for models in test directory (where they currently are)
        test_models_path = Path(__file__).parent / "models"
        
        deleted_count = 0
        
        # Delete models directory (project root)
        if models_path.exists():
            shutil.rmtree(models_path)
            print(f"Deleted models: {models_path}")
            deleted_count += 1
        else:
            print(f"Models directory not found: {models_path}")
        
        # Delete models directory (test folder)
        if test_models_path.exists():
            shutil.rmtree(test_models_path)
            print(f"Deleted test models: {test_models_path}")
            deleted_count += 1
        else:
            print(f"Test models directory not found: {test_models_path}")
        
        # Delete processed data directory
        if processed_path.exists():
            shutil.rmtree(processed_path)
            print(f"Deleted processed data: {processed_path}")
            deleted_count += 1
        else:
            print(f"Processed data directory not found: {processed_path}")
        
        # Summary
        print("=" * 40)
        if deleted_count > 0:
            print(f"Reset complete! Deleted {deleted_count} directories.")
            print("\nNext steps:")
            print("1. python src/data_generator.py")
            print("2. python src/feature_engineer.py")
            print("3. python src/model_training_orchestrator.py")
        else:
            print("Nothing to delete - already clean!")
            
    except Exception as e:
        print(f"Error during reset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
