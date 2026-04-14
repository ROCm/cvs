#!/usr/bin/env python3
"""
Script to switch your existing CVS pytest infrastructure to use the Go SSH backend.

This creates a backup of your original parallel_ssh_lib.py and replaces it with 
a version that uses the high-performance Go backend.
"""

import os
import shutil
import sys
from datetime import datetime

def switch_to_go_backend():
    """Switch to Go SSH backend"""
    
    lib_dir = "/home/amd/ichristo/cvs/cvs/lib"
    original_lib = os.path.join(lib_dir, "parallel_ssh_lib.py")
    backup_lib = os.path.join(lib_dir, f"parallel_ssh_lib.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    go_wrapper = os.path.join(lib_dir, "go_ssh_lib.py")
    
    print("🔄 Switching CVS to Go SSH backend...")
    
    # Check if files exist
    if not os.path.exists(original_lib):
        print(f"❌ Original library not found: {original_lib}")
        return False
        
    if not os.path.exists(go_wrapper):
        print(f"❌ Go wrapper not found: {go_wrapper}")
        return False
    
    # Backup original
    print(f"📦 Backing up original library to: {backup_lib}")
    shutil.copy2(original_lib, backup_lib)
    
    # Create new parallel_ssh_lib.py that imports from go_ssh_lib
    new_content = f'''"""
High-performance SSH library using Go backend.

This module provides the same API as the original parallel_ssh_lib.py
but uses a Go shared library for 20x faster SSH operations.

Original library backed up to: {os.path.basename(backup_lib)}
"""

# Import everything from the Go backend wrapper
from .go_ssh_lib import *

# For backward compatibility, make sure Pssh is available
from .go_ssh_lib import Pssh

print("🚀 Using high-performance Go SSH backend")
'''
    
    # Write new parallel_ssh_lib.py
    with open(original_lib, 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully switched to Go SSH backend!")
    print(f"📁 Original library backed up to: {backup_lib}")
    print("🚀 Your existing pytest code will now use the Go backend automatically")
    
    return True

def restore_original_backend():
    """Restore original Python backend"""
    
    lib_dir = "/home/amd/ichristo/cvs/cvs/lib"
    current_lib = os.path.join(lib_dir, "parallel_ssh_lib.py")
    
    # Find latest backup
    backups = [f for f in os.listdir(lib_dir) if f.startswith("parallel_ssh_lib.py.backup.")]
    if not backups:
        print("❌ No backup found to restore from")
        return False
    
    latest_backup = os.path.join(lib_dir, sorted(backups)[-1])
    
    print(f"🔄 Restoring original backend from: {latest_backup}")
    shutil.copy2(latest_backup, current_lib)
    print("✅ Original Python backend restored")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_original_backend()
    else:
        switch_to_go_backend()
        
        print("\n🧪 Testing the switch...")
        print("Run this to test: cd /home/amd/ichristo/cvs && python3 -c \"from cvs.lib.parallel_ssh_lib import Pssh; print('Go backend loaded successfully')\"")