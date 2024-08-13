# # animevideov3
# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe1.mp4 -n realesr-animevideov3 -s 4 --suffix outx4

# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe2.mp4 -n realesr-animevideov3 -s 4 --suffix outx4

# # realesr-general-wdn-x4v3
# realesr-general-x4v3

# # 4x_IllustrationJaNai_V1_ESRGAN_135k  load 不进去
# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe1.mp4 -n RealESRGAN_x4plus -p 0708/4x_IllustrationJaNai_V1_ESRGAN_135k -s 4 --suffix outx4

# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe2.mp4 -n RealESRGAN_x4plus -p 0708/4x_IllustrationJaNai_V1_ESRGAN_135k -s 4 --suffix outx4

# 4x-NMKD-YandereNeo-Superlite load 不进去
# CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe1.mp4 -n RealESRGAN_x4plus -p 0708/4xFSMangaV2 -s 4 --suffix outx4


# 4x-WTP-ColorDS
cd  /root/picasso/HuaqingHe/Real-ESRGAN
CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video-time.py -i inputs/video/wangzhe1.mp4 -n RealESRGAN_x4plus -p 0708/4x-WTP-ColorDS -s 4 --suffix outx4

CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe2.mp4 -n RealESRGAN_x4plus -p 0708/4x-WTP-ColorDS -s 4 --suffix outx4
