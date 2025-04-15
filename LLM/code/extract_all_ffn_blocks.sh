#!/bin/bash
# filepath: extract_all_ffn_blocks.sh

# Create output directory if it doesn't exist
mkdir -p models_1.5B_f16

# Loop through all blocks from 0 to 27
for i in {0..27}
do
    echo "Processing block $i..."
    python ffn.py --embed_dim 1536 --hidden_dim 8960 --load_weights \
        --down_proj_weights weights_1.5B_f16/blk.$i.ffn_down.weight.npy \
        --gate_proj_weights weights_1.5B_f16/blk.$i.ffn_gate.weight.npy \
        --up_proj_weights weights_1.5B_f16/blk.$i.ffn_up.weight.npy \
        --dtype float16 \
        --save_path models_1.5B_f16_seq1/blk.$i.ffn
    
    echo "Block $i complete!"
done

echo "All blocks processed successfully!"