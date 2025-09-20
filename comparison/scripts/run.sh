
MODELS=('npnet' 'pointnn' 'pointgn')

for MODEL in "${MODELS[@]}"; do
    python tasks/eval_modelnet.py --model "$MODEL"
done


MODELS=('npnet' 'pointnn')

for MODEL in "${MODELS[@]}"; do
    python tasks/eval_shapenet.py --model "$MODEL"
done

