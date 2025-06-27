#!/usr/bin/env python3
"""
Script to fix common mypy type annotation issues in the HybridSuperQubits package.
"""

import re
import os
from pathlib import Path

def fix_optional_imports_and_annotations(file_path):
    """Add Optional import and fix parameter annotations."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if Optional is already imported
    has_optional = 'Optional' in content and 'from typing import' in content
    
    # Add Optional to imports if needed
    if not has_optional and '= None' in content:
        # Find the typing import line
        typing_pattern = r'from typing import (.+)'
        match = re.search(typing_pattern, content)
        if match:
            imports = match.group(1)
            if 'Optional' not in imports:
                new_imports = imports.rstrip() + ', Optional'
                content = re.sub(typing_pattern, f'from typing import {new_imports}', content)
        else:
            # Add new typing import after other imports
            import_end = 0
            for line_num, line in enumerate(content.split('\n')):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = line_num
            
            lines = content.split('\n')
            lines.insert(import_end + 1, 'from typing import Optional')
            content = '\n'.join(lines)
    
    # Fix parameter annotations
    patterns_to_fix = [
        # Standard parameter patterns
        (r'(\w+): (np\.ndarray|str|int|float|dict\[.*?\]|list\[.*?\]|tuple\[.*?\]|Union\[.*?\]|SpectrumData|Callable\[.*?\]) = None', 
         r'\1: Optional[\2] = None'),
        # More complex patterns
        (r'(\w+): (tuple\[ndarray\[Any, Any\], ndarray\[Any, Any\]\]) = None',
         r'\1: Optional[\2] = None'),
    ]
    
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)
    
    # Fix specific problematic patterns
    # Fix bool with str annotation
    content = re.sub(r'rotate: str = False', 'rotate: bool = False', content)
    
    return content

def fix_return_statements(file_path):
    """Fix NotImplementedError returns to raises."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix return NotImplementedError
    content = re.sub(r'return NotImplementedError\((.*?)\)', r'raise NotImplementedError(\1)', content)
    
    return content

def add_else_clauses_for_exhaustive_branches(file_path):
    """Add else clauses to functions that have exhaustive if-elif but mypy doesn't know."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # This is a bit complex to do with regex, so let's handle specific known cases
    if 'ferbo.py' in str(file_path):
        # Look for flux_grouping patterns that need else clauses
        patterns = [
            (r'(if self\.flux_grouping == "EL":.*?elif self\.flux_grouping == "ABS":.*?return.*?)(\n\n|\n    def)', 
             r'\1\n        else:\n            raise ValueError(f"Unknown flux_grouping: {self.flux_grouping}")\2')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    return content

def process_file(file_path):
    """Process a single Python file to fix type annotations."""
    print(f"Processing {file_path}")
    
    # Read original content
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    # Apply fixes
    content = original_content
    content = fix_optional_imports_and_annotations(file_path)
    content = fix_return_statements(file_path)
    content = add_else_clauses_for_exhaustive_branches(file_path)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  Updated {file_path}")
    else:
        print(f"  No changes needed for {file_path}")

def main():
    """Fix type annotations in all Python files in the HybridSuperQubits package."""
    package_dir = Path("HybridSuperQubits")
    
    for py_file in package_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            process_file(py_file)

if __name__ == "__main__":
    main()
