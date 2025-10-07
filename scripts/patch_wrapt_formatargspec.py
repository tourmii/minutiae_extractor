#!/usr/bin/env python3
"""
Patch script to fix wrapt library's formatargspec import issue on Python 3.11+.

This script adds a compatibility shim to wrapt/decorators.py to provide
formatargspec when it's not available in the inspect module (Python 3.11+).

Usage:
    python scripts/patch_wrapt_formatargspec.py
"""

import sys
import os
from pathlib import Path


def find_wrapt_decorators():
    """Find the wrapt/decorators.py file in the current Python environment."""
    try:
        import wrapt
        wrapt_path = Path(wrapt.__file__).parent / "decorators.py"
        if wrapt_path.exists():
            return wrapt_path
    except ImportError:
        pass
    
    # Fallback: search in sys.path
    for path in sys.path:
        candidate = Path(path) / "wrapt" / "decorators.py"
        if candidate.exists():
            return candidate
    
    return None


def check_if_already_patched(file_path):
    """Check if the file has already been patched."""
    content = file_path.read_text()
    return "# COMPATIBILITY SHIM FOR PYTHON 3.11+" in content


def patch_wrapt_decorators(file_path):
    """Apply the formatargspec compatibility shim to wrapt/decorators.py."""
    print(f"Reading {file_path}...")
    content = file_path.read_text()
    
    # Check if already patched
    if check_if_already_patched(file_path):
        print("✓ File is already patched. No changes needed.")
        return True
    
    # Find the import line we need to patch after
    import_line = "from inspect import ismethod, isclass, formatargspec"
    
    if import_line not in content:
        print("✗ Could not find the expected import line. The wrapt version may be different.")
        print("Expected: from inspect import ismethod, isclass, formatargspec")
        return False
    
    # Create the patch
    patch = """from inspect import ismethod, isclass, formatargspec

# COMPATIBILITY SHIM FOR PYTHON 3.11+
# formatargspec was removed in Python 3.11, but older wrapt versions still need it
if not hasattr(sys.modules['inspect'], 'formatargspec'):
    import inspect as _inspect_module
    
    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                     formatarg=str,
                     formatvarargs=lambda name: '*' + name,
                     formatvarkw=lambda name: '**' + name,
                     formatvalue=lambda value: '=' + repr(value)):
        '''Format an argument spec from the values returned by inspect.getfullargspec.'''
        specs = []
        if defaults:
            firstdefault = len(args) - len(defaults)
        else:
            firstdefault = -1
        for i, arg in enumerate(args):
            spec = formatarg(arg)
            if defaults and i >= firstdefault:
                spec = spec + formatvalue(defaults[i - firstdefault])
            specs.append(spec)
        if varargs is not None:
            specs.append(formatvarargs(varargs))
        if varkw is not None:
            specs.append(formatvarkw(varkw))
        return '(' + ', '.join(specs) + ')'
    
    _inspect_module.formatargspec = formatargspec
    formatargspec = _inspect_module.formatargspec
# END COMPATIBILITY SHIM"""
    
    # Replace the import line with the patched version
    patched_content = content.replace(import_line, patch)
    
    # Backup the original file
    backup_path = file_path.with_suffix('.py.bak')
    print(f"Creating backup at {backup_path}...")
    file_path.rename(backup_path)
    
    # Write the patched content
    print(f"Writing patched file to {file_path}...")
    file_path.write_text(patched_content)
    
    print("✓ Successfully patched wrapt/decorators.py!")
    print(f"  Original backed up to: {backup_path}")
    return True


def main():
    print("=" * 70)
    print("WRAPT FORMATARGSPEC COMPATIBILITY PATCH")
    print("=" * 70)
    print()
    
    # Find the wrapt decorators file
    wrapt_file = find_wrapt_decorators()
    
    if not wrapt_file:
        print("✗ Could not find wrapt/decorators.py in the current Python environment.")
        print("  Make sure wrapt is installed: pip install wrapt")
        return 1
    
    print(f"Found wrapt at: {wrapt_file}")
    print()
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version < (3, 11):
        print("✓ Python version is < 3.11. Patch may not be needed, but will apply anyway.")
    else:
        print("✓ Python version is 3.11+. Patch is needed.")
    print()
    
    # Apply the patch
    success = patch_wrapt_decorators(wrapt_file)
    
    if success:
        print()
        print("=" * 70)
        print("PATCH COMPLETE")
        print("=" * 70)
        print()
        print("You can now import wrapt (and dependencies like pylint) without errors.")
        print()
        print("To verify, try:")
        print("  python -c 'from wrapt import decorators; print(\"Success!\")'")
        return 0
    else:
        print()
        print("=" * 70)
        print("PATCH FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
