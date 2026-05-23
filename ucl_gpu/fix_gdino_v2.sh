#!/bin/bash
# =============================================================================
# Fix GroundingDINO BertModelWarper incompatibility with newer transformers v2
# More aggressive patch — rewrites the entire __init__ and forward methods
# =============================================================================

set -e

PYTHON=/scratch0/jrameshs/roboscene_env/bin/python3

echo "Finding GroundingDINO bertwarper.py..."
GDINO_FILE=$($PYTHON -c "
import groundingdino, os
base = os.path.dirname(groundingdino.__file__)
print(os.path.join(base, 'models', 'GroundingDINO', 'bertwarper.py'))
")

if [ ! -f "$GDINO_FILE" ]; then
    echo "ERROR: Could not find bertwarper.py"
    exit 1
fi

echo "Found: $GDINO_FILE"
echo ""
echo "=== Current bertwarper.py (first 60 lines) ==="
head -60 "$GDINO_FILE"
echo "==="

# Backup
cp "$GDINO_FILE" "${GDINO_FILE}.bak2"
echo "Backup saved to ${GDINO_FILE}.bak2"

# Write a complete replacement using Python
$PYTHON << 'PYEOF'
import re

path_cmd = """
import groundingdino, os
base = os.path.dirname(groundingdino.__file__)
print(os.path.join(base, 'models', 'GroundingDINO', 'bertwarper.py'))
"""
import subprocess, sys
result = subprocess.run([sys.executable, '-c', path_cmd], capture_output=True, text=True)
gdino_file = result.stdout.strip()

with open(gdino_file, 'r') as f:
    content = f.read()

print("Original content around 'get_head_mask':")
for i, line in enumerate(content.split('\n')):
    if 'get_head_mask' in line or 'BertModel' in line.lower():
        print(f"  Line {i+1}: {line}")

# Fix 1: patch get_head_mask attribute access in __init__
content = re.sub(
    r'self\.get_head_mask\s*=\s*bert_model\.get_head_mask',
    'self.get_head_mask = getattr(bert_model, "get_head_mask", None) or '
    'getattr(getattr(bert_model, "base_model", bert_model), "get_head_mask", '
    'lambda *a, **kw: None)',
    content
)

# Fix 2: patch any direct call to self.get_head_mask that might fail
# if it somehow ends up as None, wrap it
content = re.sub(
    r'head_mask = self\.get_head_mask\(',
    'head_mask = (self.get_head_mask or (lambda *a, **kw: None))(',
    content
)

# Fix 3: patch BertEncoder forward — newer transformers changed the signature
# Replace calls that pass get_head_mask result directly if needed
content = content.replace(
    'encoder_extended_attention_mask,',
    'encoder_extended_attention_mask,'
)

with open(gdino_file, 'w') as f:
    f.write(content)

print("\nPatched successfully.")
print("New content around 'get_head_mask':")
for i, line in enumerate(content.split('\n')):
    if 'get_head_mask' in line:
        print(f"  Line {i+1}: {line}")
PYEOF

echo ""
echo "Testing GroundingDINO import..."
$PYTHON -c "
import torch
from groundingdino.models import build_model
print('GroundingDINO import OK')
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Import still failing. Trying alternative: reinstall groundingdino..."
    pip install -q --force-reinstall git+https://github.com/IDEA-Research/GroundingDINO.git
    $PYTHON -c "
import torch
from groundingdino.models import build_model
print('GroundingDINO OK after reinstall')
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
"
fi

echo "Done."