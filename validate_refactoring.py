#!/usr/bin/env python3
"""
Validation script to confirm successful DBC → Alexandria refactoring.

This script checks that:
1. No legacy 'dbc_unified_content' references remain
2. All services use 'alexandria_books' collection consistently
3. Configuration is properly unified
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

def search_files_for_pattern(directory: Path, pattern: str, extensions: List[str] = None) -> List[Tuple[str, int, str]]:
    """Search for pattern in files and return matches."""
    matches = []
    extensions = extensions or ['.py']
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            # Skip virtual environment and build directories
            if any(skip_dir in file_path.parts for skip_dir in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            matches.append((str(file_path.relative_to(directory)), line_num, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue
    
    return matches

def validate_no_legacy_references() -> bool:
    """Check that no legacy DBC references remain."""
    print("🔍 Checking for legacy 'dbc_unified_content' references...")
    
    project_root = Path(__file__).parent / 'src'
    matches = search_files_for_pattern(project_root, r'dbc_unified_content')
    
    if matches:
        print("❌ Found legacy 'dbc_unified_content' references:")
        for file_path, line_num, line in matches:
            print(f"   {file_path}:{line_num} - {line}")
        return False
    else:
        print("✅ No legacy 'dbc_unified_content' references found")
        return True

def validate_collection_name_consistency() -> bool:
    """Check that alexandria_books is used consistently."""
    print("\n🔍 Checking for 'alexandria_books' usage...")
    
    project_root = Path(__file__).parent / 'src'
    
    # Check that DEFAULT_COLLECTION_NAME is defined
    config_file = project_root / 'utils' / 'config.py'
    if not config_file.exists():
        print("❌ Config file not found")
        return False
    
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    if 'DEFAULT_COLLECTION_NAME = "alexandria_books"' not in config_content:
        print("❌ DEFAULT_COLLECTION_NAME not properly defined")
        return False
    
    print("✅ DEFAULT_COLLECTION_NAME properly defined as 'alexandria_books'")
    
    # Check for proper usage in key files
    key_files = [
        'utils/enhanced_database.py',
        'services/enhanced_embedding_service.py',
        'utils/performance_tester.py'
    ]
    
    all_correct = True
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            
            if 'DEFAULT_COLLECTION_NAME' in content:
                print(f"✅ {file_path} uses DEFAULT_COLLECTION_NAME")
            else:
                print(f"❌ {file_path} doesn't import DEFAULT_COLLECTION_NAME")
                all_correct = False
    
    return all_correct

def validate_no_dbc_branding() -> bool:
    """Check that DBC branding has been replaced with Alexandria."""
    print("\n🔍 Checking for remaining DBC branding...")
    
    project_root = Path(__file__).parent / 'src'
    
    # Look for "Dynamic Book Companion" and "DBC" in key places
    problematic_patterns = [
        r'Dynamic Book Companion',
        r'DBC Application',
        r'DBC Platform',
        r'dbc\.log',
        r'dbc-mcp-server'
    ]
    
    all_clean = True
    for pattern in problematic_patterns:
        matches = search_files_for_pattern(project_root, pattern)
        if matches:
            print(f"⚠️  Found '{pattern}' references:")
            for file_path, line_num, line in matches[:3]:  # Show first 3 matches
                print(f"   {file_path}:{line_num} - {line}")
            if len(matches) > 3:
                print(f"   ... and {len(matches) - 3} more")
            all_clean = False
    
    if all_clean:
        print("✅ No problematic DBC branding found")
    
    return all_clean

def validate_test_files() -> bool:
    """Check that test files use the correct collection name."""
    print("\n🔍 Checking test files...")
    
    project_root = Path(__file__).parent
    test_files = list(project_root.glob('test*.py'))
    
    all_correct = True
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
        
        if 'dbc_unified_content' in content:
            print(f"❌ {test_file.name} still uses 'dbc_unified_content'")
            all_correct = False
        elif 'alexandria_books' in content:
            print(f"✅ {test_file.name} uses 'alexandria_books'")
    
    if not test_files:
        print("📝 No test files found")
    
    return all_correct

def print_collection_usage_summary():
    """Print a summary of collection name usage."""
    print("\n📊 Collection Name Usage Summary:")
    
    project_root = Path(__file__).parent / 'src'
    
    # Count usage of different collection references
    usage_counts = {
        'DEFAULT_COLLECTION_NAME': 0,
        'alexandria_books': 0,
        'settings.chroma_collection_name': 0
    }
    
    for pattern, count_key in [
        (r'DEFAULT_COLLECTION_NAME', 'DEFAULT_COLLECTION_NAME'),
        (r'"alexandria_books"', 'alexandria_books'),
        (r'settings\.chroma_collection_name', 'settings.chroma_collection_name')
    ]:
        matches = search_files_for_pattern(project_root, pattern)
        usage_counts[count_key] = len(matches)
    
    for usage_type, count in usage_counts.items():
        print(f"   {usage_type}: {count} occurrences")

def main():
    """Run all validation checks."""
    print("🚀 Alexandria Refactoring Validation")
    print("=" * 50)
    
    checks = [
        ("Legacy References", validate_no_legacy_references),
        ("Collection Consistency", validate_collection_name_consistency),
        ("DBC Branding", validate_no_dbc_branding),
        ("Test Files", validate_test_files)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
    
    print_collection_usage_summary()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 Validation Summary:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All validations passed!")
        print(f"✅ DBC → Alexandria refactoring completed successfully")
        print(f"🏆 Collection naming unified to 'alexandria_books'")
        print(f"🔄 Ingestion and query systems now use same collection")
    else:
        print(f"\n⚠️  Some validations failed - see details above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)