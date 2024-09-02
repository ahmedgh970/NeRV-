# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

# To continue training from ckpts just add --weight ... in the train script
if __name__ == '__main__':

    # ------------------------------------
    # --- bunny
    # ------------------------------------
    # --------------------------- NeRV++ -----------------------------------------------
    # --- Train and prune NeRV++ XS (--fc_hw_dim 9_16_8)
    os.system("python train.py -e 300  --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                   --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_8 --expansion 1  \
                   --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2 --conv_type conv \
                   -b 1  --lr 0.0005 --norm none --act swish")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                   --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_8 --expansion 1  \
                   --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
                   -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
                   --weight output/nerv_plus/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_8__exp1.0_reduce2_low80_blk1_cycle1_gap1_e300_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_latest.pth --not_resume_epoch --prune_ratio 0.4")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                   --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_8 --expansion 1  \
                   --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
                   -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
                   --weight output/nerv_plus/prune_ab/bunny/embed1.25_40_512_1_fc_9_16_8__exp1.0_reduce2_low80_blk1_cycle1_gap1_e100_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_Prune0.4_0.0_actswish_107/model_latest.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 1")
    

    # --- Train and prune NeRV++ S (--fc_hw_dim 9_16_26)
    os.system("python train.py -e 300  --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                       --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
                       --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2 --conv_type conv \
                       -b 1  --lr 0.0005 --norm none --act swish")
    

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                       --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
                       --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
                       -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
                       --weight output/nerv_plus/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low80_blk1_cycle1_gap1_e300_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_latest.pth --not_resume_epoch --prune_ratio 0.4")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
                   --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
                   --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
                   -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
                   --weight output/nerv_plus/prune_ab/bunny/embed1.25_40_512_1_fc_9_16_26__exp1.0_reduce2_low80_blk1_cycle1_gap1_e100_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_Prune0.4_0.0_actswish_107/model_latest.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 1")

    # --- Train and prune NeRV++ M (--fc_hw_dim 9_16_58)
    os.system("python train.py -e 300  --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_58 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2 --conv_type conv \
               -b 1  --lr 0.0005 --norm none --act swish")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_58 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
               -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
               --weight output/nerv_plus/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_58__exp1.0_reduce2_low80_blk1_cycle1_gap1_e300_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_latest.pth --not_resume_epoch --prune_ratio 0.4")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_58 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
               -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
               --weight output/nerv_plus/prune_ab/bunny/embed1.25_40_512_1_fc_9_16_58__exp1.0_reduce2_low80_blk1_cycle1_gap1_e100_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_Prune0.4_0.0_actswish_107/model_latest.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 1")

    # --- Train and prune NeRV++ L (--fc_hw_dim 9_16_112)
    os.system("python train.py -e 300  --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2 --conv_type conv \
               -b 1  --lr 0.0005 --norm none --act swish")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
               -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
               --weight output/nerv_plus/bunny_ab/bunny/embed1.25_40_512_1_fc_9_16_112__exp1.0_reduce2_low80_blk1_cycle1_gap1_e300_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_actswish_/model_latest.pth --not_resume_epoch --prune_ratio 0.4")

    os.system("python train.py -e 100   --lower-width 80 --num-blocks 1 --dataset bunny --frame_gap 1 \
               --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
               --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
               -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
               --weight output/nerv_plus/prune_ab/bunny/embed1.25_40_512_1_fc_9_16_112__exp1.0_reduce2_low80_blk1_cycle1_gap1_e100_warm0_b1_conv_lr0.0005_cosine_Fusion6_Strd5,2,2,2,2_SinRes_Prune0.4_0.0_actswish_107/model_latest.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 1")
