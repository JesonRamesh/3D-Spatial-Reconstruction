#!/bin/bash
# =============================================================================
# Fix GroundingDINO BertModelWarper incompatibility with newer transformers
# =============================================================================
# The error:
#   AttributeError: 'BertModel' object has no attribute 'get_head_mask'
# Root cause:
#   groundingdino/models/GroundingDINO/bertwarper.py line 29 tries to grab
#   get_head_mask as an instance attribute. In transformers>=4.36 this method
#   moved to the base class and is no longer directly on the instance.
# Fix:
#   Wrap the attribute access in a getattr with fallback to base_model.
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
    echo "ERROR: Could not find bertwarper.py at $GDINO_FILE"
    exit 1
fi

echo "Found: $GDINO_FILE"
echo "Current line 29:"
sed -n '27,32p' "$GDINO_FILE"

# Check if already patched
if grep -q 'get_head_mask.*getattr\|hasattr.*get_head_mask' "$GDINO_FILE"; then
    echo "Already patched. Nothing to do."
else
    echo "Patching..."
    # Backup original
    cp "$GDINO_FILE" "${GDINO_FILE}.bak"
    
    # Replace the problematic line
    sed -i 's/self\.get_head_mask = bert_model\.get_head_mask/self.get_head_mask = getattr(bert_model, "get_head_mask", None) or getattr(bert_model.base_model, "get_head_mask", None) or (lambda *a, **k: None)/' "$GDINO_FILE"
    
    echo "Patched. New line:"
    sed -n '27,32p' "$GDINO_FILE"
fi

# Test
echo "Testing GroundingDINO import..."
$PYTHON -c "
import torch
from groundingdino.models import build_model
print('GroundingDINO OK')
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
"

echo "Done."