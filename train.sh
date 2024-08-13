sdf train create hehqesrganv5-4gpu --worker 1 --image lab/all_in_one:cl4_dreambooth_xformers_cuda11.3_cuda11.7_sdwebui --cmd 'cd /root/picasso/HuaqingHe/ && bash sleep.sh' --cpu 40 --mem 100 --gpu 4 --gputype 3090 --priority high
sdf train create hehqesrganv5-10804gpu --worker 1 --image lab/all_in_one:cl4_dreambooth_xformers_cuda11.3_cuda11.7_sdwebui --cmd 'cd /root/picasso/HuaqingHe/ && bash sleep.sh' --cpu 40 --mem 100 --gpu 4 --gputype 1080ti --priority high
scp -P 30859 -r root@10.255.0.125:/simple/HuaqingHe/.cache/.cache /root
scp -P 30859 -r root@10.255.0.125:/simple/HuaqingHe/conda/envs/Real-ESRGAN /opt/conda/envs
94B6O39Y

# step 1 multi scale resize
python scripts/generate_multiscale_DF2K.py --input /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/APISR_datasets/APISR_720p_4xcrop \
--output /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/Real-ESRGAN_datasets/Wangzhe/Wangzhe_multiscale

# step 2 crop subimage
python scripts/extract_subimages.py --input /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/Real-ESRGAN_datasets/Wangzhe/Wangzhe_multiscale \
--output /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/Real-ESRGAN_datasets/Wangzhe/Wangzhe_multiscale_sub --crop_size 400 --step 200

# step3 name txt
python scripts/generate_meta_info.py --input /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/APISR_datasets/APISR_720p_4xcrop, /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/Real-ESRGAN_datasets/Wangzhe/Wangzhe_multiscale_sub --root /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training, /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training --meta_info /root/cloud/cephfs-group-hdvideo_group/Datasets/Game_training/Real-ESRGAN_datasets/Wangzhe/meta_info/meta_info_Wangzhe_hr+multiscalesub.txt

# step4 train L1 -debug
python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug

# step5 train L1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --auto_resume


## finetune
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_UDS_x4plus.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4wangzhe-v3-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4wangzhe-v4.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=1 python realesrgan/train.py  -opt options/finetune_realesrgan_x4wangzhe-v4.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x2wangzhe-v4-deblur-addaddnoise.yml --launcher pytorch --auto_resume

# for video compression
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-deblur-compression.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compression-only.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compressionless-blur.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compressionless-deblur-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compressionhard-addblur-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compressionhard-addblurnoisefinal-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x4wangzhe-v5-compressionless-blurmore-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_L1_gan_x4wangzhe-v5-clesser-blurmore-1080.yml --launcher pytorch --auto_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_L1_gan_x4wangzhe-v5-clesser-blurmore-1080-wgan.yml --launcher pytorch --auto_resume

# pair data
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_WXV1_x4plus_pairdata.yml --launcher pytorch --auto_resume


# x2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py \
-opt options/finetune_realesrgan_x2wangzhe-v5-compreessionless_addnoise_deblur.yml --launcher pytorch --auto_resume

torchrun --nproc_per_node=4 realesrgan/train.py \
-opt options/finetune_realesrgan_x2wangzhe-v5-compreessionless_addnoise_deblur.yml --launcher pytorch --auto_resume

# debug
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/Real-ESRGAN/bin/python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 realesrgan/train.py \
-opt options/finetune-compressionhard-addblurfinal-debugforlr.yml --auto_resume
