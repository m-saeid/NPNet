
# MODELNET40 Evaluation
DIMS=(5 10 15 20 25 30 35 40 45 50 65 70 100 150)
KS=(100 102 104 106 108 110 112 114 116 118 120 122 124 126)
STAGES=(2 3 4)


for DIM in "${DIMS[@]}"; do
    python tasks/eval_modelnet_ablation.py --model 'npnet' --dim "$DIM"
done

for K in "${KS[@]}"; do
    python tasks/eval_modelnet_ablation.py --model 'npnet' --k "$K"
done

for STAGE in "${STAGES[@]}"; do
    python tasks/eval_modelnet_ablation.py --model 'npnet' --stage "$STAGE"
done


# SHAPENET Evaluation
DIMS=(12 16 20 24 28 32 36 40 44 48 60 72 84 96 108 120 132 144 156 168 180 192 204 216 228 240 252 264 276 288 300 312 324 336 348 360 372 384 396 400)
KS=(40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120)
STAGES=(1 2 3 4)

for DIM in "${DIMS[@]}"; do
    python tasks/eval_shapenet_ablation.py --model 'npnet' --dim "$DIM"
done

for K in "${KS[@]}"; do
    python tasks/eval_shapenet_ablation.py --model 'npnet' --k "$K"
done

for STAGE in "${STAGES[@]}"; do
    python tasks/eval_shapenet_ablation.py --model 'npnet' --stage "$STAGE"
done