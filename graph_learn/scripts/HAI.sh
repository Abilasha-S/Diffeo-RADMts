gpu_n=$1
DATASET=$2

seed=5
BATCH_SIZE=128
#SLIDE_WIN=5
SLIDE_WIN=100
#dim=64
dim=100 ##GNN layer out dim same as slide_win for diff
out_layer_num=1
SLIDE_STRIDE=1
topk=1
out_layer_inter_dim=128
val_ratio=0.2
decay=0
lamda=0.0001
a_init=2
weight_p=1
weight_r=1




path_pattern="${DATASET}"
COMMENT="${DATASET}"

EPOCH=20
report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
	    -lamda $lamda \
        -weight_p $weight_p \
        -weight_r $weight_r \
        -a_init $a_init \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
	    -lamda $lamda \
        -weight_p $weight_p \
        -weight_r $weight_r \
        -a_init $a_init
fi


