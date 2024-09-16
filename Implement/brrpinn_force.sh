#python program running
python force.py \
    --lr 0.0002 \
    --batch_size 1024 \
    --epochs 1000000 \
    --gpu True \
    --train_rec_size 128 \
    --train_gen_random True \
    --weight_equ1 7\
    --weight_equ2 7\
    --weight_equ3 15\
    --weight_equ4 5\
    --weight_equ5 15\
    --weight_equ6 100\
    --brr 40 \
    --center_value 1\
    --network_MLP "(128,128,128,128,128)" \
    --check_every 1000 \
    --save_dict "train_result/brrpinn_force"\
    --impose 1





                    
