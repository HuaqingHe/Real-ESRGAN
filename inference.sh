# single gpu and single process inference
CUDA_VISIBLE_DEVICES=0 python inference_realesrgan_video.py -i inputs/video/onepiece_demo.mp4 -n realesr-animevideov3 -s 4 --suffix outx4
CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhetest.mov -n realesr-animevideov3 -s 4 --suffix outx4
CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe1.mp4 -n realesr-animevideov3 -s 4 --suffix outx4
CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe2.mp4 -n realesr-animevideov3 -s 4 --suffix outx4

CUDA_VISIBLE_DEVICES=1 python inference_realesrgan_video.py -i inputs/video/wangzhe2.mp4 -n RealESRGAN_x4plus_anime_6B -p 4x-UltraSharp -s 4 --suffix outx4

# single gpu and multi process inference (you can use multi-processing to improve GPU utilization)
CUDA_VISIBLE_DEVICES=0 python inference_realesrgan_video.py -i inputs/video/onepiece_demo.mp4 -n realesr-animevideov3 -s 2 --suffix outx2 --num_process_per_gpu 2
# multi gpu and multi process inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_realesrgan_video.py -i inputs/video/onepiece_demo.mp4 -n realesr-animevideov3 -s 2 --suffix outx2 --num_process_per_gpu 2