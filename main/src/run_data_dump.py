"""
Run PDF data dumping pipeline from src folder
Usage: python run_data_dump.py

This script loads all configuration from constants.py
"""

import subprocess
import sys
from pathlib import Path

# Import constants to verify paths
from constants import PDFS_DIR, OUTPUT_DIR, verify_paths

def main():
    """Run the data dump pipeline"""
    
    # Verify paths exist
    print("📋 Verifying configuration...")
    verify_paths()
    print(f"✅ Config verified")
    print(f"   PDFs: {PDFS_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # Get data_dump script path
    script_path = Path(__file__).parent / "data_dump" / "dump_data.py"
    
    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        return 1
    
    print(f"\n🚀 Running data dump pipeline from: {Path(__file__).parent}")
    print(f"📄 Script: {script_path.name}")
    print("-" * 60)
    
    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
