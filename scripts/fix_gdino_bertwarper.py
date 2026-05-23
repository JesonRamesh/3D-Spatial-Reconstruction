#!/usr/bin/env python3
import sys, os, re, argparse


def find_bertwarper():
    try:
        import groundingdino
        base = os.path.dirname(groundingdino.__file__)
        path = os.path.join(base, 'models', 'GroundingDINO', 'bertwarper.py')
        if os.path.exists(path):
            return path
    except ImportError:
        pass
    return None


def show_file(path):
    print(f'\n=== {path} ===')
    with open(path) as f:
        for i, line in enumerate(f, 1):
            print(f'{i:4d}: {line}', end='')
    print('\n===')


def patch_bertwarper(path):
    with open(path, 'r') as f:
        original = f.read()

    print(f'Patching: {path}\n')

    lines = original.split('\n')
    print('Lines containing get_head_mask before patch:')
    for i, line in enumerate(lines, 1):
        if 'get_head_mask' in line:
            print(f'  {i:4d}: {line}')
    print()

    patched = original

    # Patch 1: __init__ attribute assignment
    patched = re.sub(
        r'self\.get_head_mask\s*=\s*bert_model\.get_head_mask',
        (
            'self.get_head_mask = ('
            'getattr(bert_model, "get_head_mask", None) or '
            'getattr(getattr(bert_model, "base_model", None), "get_head_mask", None) or '
            '(lambda *a, **kw: None)'
            ')'
        ),
        patched
    )

    # Patch 2: guard any call to self.get_head_mask against None
    patched = re.sub(
        r'\bself\.get_head_mask\(',
        '(self.get_head_mask or (lambda *a, **kw: None))(',
        patched
    )

    if patched == original:
        print('WARNING: No changes made - pattern not found in file.')
        print('Showing all lines with mask:')
        for i, line in enumerate(lines, 1):
            if 'mask' in line.lower():
                print(f'  {i:4d}: {line}')
    else:
        backup = path + '.bak'
        with open(backup, 'w') as f:
            f.write(original)
        print(f'Backup saved: {backup}')
        with open(path, 'w') as f:
            f.write(patched)
        print('Patch applied.')

    print('\nLines containing get_head_mask after patch:')
    for i, line in enumerate(patched.split('\n'), 1):
        if 'get_head_mask' in line:
            print(f'  {i:4d}: {line}')

    return patched != original


def test_import():
    print('\nTesting GroundingDINO import...')
    try:
        import torch
        from groundingdino.models import build_model
        print('  GroundingDINO import: OK')
        if torch.cuda.is_available():
            print(f'  GPU: {torch.cuda.get_device_name(0)}')
        else:
            print('  GPU: not available (CPU mode)')
        return True
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix GroundingDINO bertwarper.py')
    parser.add_argument('--show', action='store_true', help='Print full bertwarper.py')
    args = parser.parse_args()

    path = find_bertwarper()
    if path is None:
        print('ERROR: groundingdino not installed or bertwarper.py not found.')
        print('Install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git')
        sys.exit(1)

    if args.show:
        show_file(path)

    patch_bertwarper(path)
    ok = test_import()

    if not ok:
        print('\nImport still failing. Showing full bertwarper.py:')
        show_file(path)
        sys.exit(1)
    else:
        print('\nGroundingDINO OK')


if __name__ == '__main__':
    main()
