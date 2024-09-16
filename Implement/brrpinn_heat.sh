#python program running
python heat.py \
    --lr 0.00001 \
    --batch_size 1024 \
    --epochs 1000000 \
    --gpu True \
    --train_rec_size 128 \
    --train_bound_size 64 \
    --train_gen_random True \
    --weight_equ1 1\
    --weight_equ2 5\
    --weight_equ3 2\
    --weight_equ4 5\
    --brr 30 \
    --center_value 1\
    --network_MLP "(32,32,32,32,32)" \
    --check_every 1000 \
    --save_dict "train_result/brrpinn_heat"\
    --impose 1





                    
