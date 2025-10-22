#!/usr/bin/env python3
"""
Master script to generate all publication outputs.
Run this script to regenerate all figures and tables.
"""

import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent

def run_script(script_name):
    """Run a Python script and report status."""
    script_path = ANALYSIS_DIR / script_name
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print('='*80)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error code {e.returncode}")
        return False

def main():
    print("="*80)
    print("GENERATING ALL PUBLICATION OUTPUTS")
    print("="*80)
    print("\nThis will generate:")
    print("  - Main text figures (2)")
    print("  - Appendix figures (2+)")
    print("  - Main text tables (2)")
    print("  - Appendix tables (2+)")
    print("\nAll outputs in EPS and PNG formats")
    print("="*80)

    scripts = [
        'generate_figures.py',
        'generate_tables.py',
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)

    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {script:30s} {status}")

    if all(results.values()):
        print("\n🎉 ALL OUTPUTS GENERATED SUCCESSFULLY!")
        print("\nOutput locations:")
        print(f"  Figures: {ANALYSIS_DIR.parent / 'figures'}")
        print(f"  Tables:  {ANALYSIS_DIR.parent / 'tables'}")
    else:
        print("\n⚠️  Some scripts failed. Check error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
