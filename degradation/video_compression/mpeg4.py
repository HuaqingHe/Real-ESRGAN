import torch, sys, os, random
import cv2
import shutil
import numpy as np

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder

# Define the chroma subsampling factor (e.g., 2 for 4:2:0)
chroma_subsampling_factor = 2
# Ensure the size is a multiple of the chroma subsampling factor
def adjust_size(width, height, factor, min_size=64):
    adjusted_width = max((width // factor) * factor, min_size)
    adjusted_height = max((height // factor) * factor, min_size)
    return adjusted_width, adjusted_height

class MPEG4():
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    def compress_and_store(self, opt, single_frame, idx=0, tmp_path='tmp'):
        ''' Compress and Store the whole batch as MPEG-4 (for 2nd stage)
        Args:
            single_frame (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path
            idx (int):              A unique process idx
        Return:
            None
        '''

        # Prepare
        # temp_input_path = "tmp_mpeg4/input"
        # video_store_dir = "tmp_mpeg4/encoded.mp4"
        # temp_store_path = "tmp_mpeg4/output"
        # if os.path.exists(temp_input_path):
        #     shutil.rmtree(temp_input_path)
        # os.makedirs(temp_input_path)
        # if os.path.exists(temp_store_path):
        #     shutil.rmtree(temp_store_path)
        # os.makedirs(temp_store_path)
        temp_input_path = tmp_path+"/input_"+str(idx)
        video_store_dir = tmp_path+"/encoded_"+str(idx)+".mp4"
        temp_store_path = tmp_path+"/output_"+str(idx)
        # if os.path.exists(temp_input_path):
        #     shutil.rmtree(temp_input_path)
        # if os.path.exists(temp_store_path):
        #     shutil.rmtree(temp_store_path)
        # if os.path.exists(video_store_dir):
        #     shutil.rmtree(video_store_dir)
        os.makedirs(temp_input_path)
        os.makedirs(temp_store_path)

        # Move frame
        single_frame = cv2.cvtColor(single_frame, cv2.COLOR_BGR2RGB)
        # Adjust size to be a multiple of the chroma subsampling factor
        ori_h, ori_w = single_frame.shape[0:2]
        adjusted_w, adjusted_h = adjust_size(ori_w, ori_h, chroma_subsampling_factor)
        if adjusted_w != ori_w or adjusted_h != ori_h:
            single_frame = cv2.resize(single_frame, (adjusted_w, adjusted_h))
        cv2.imwrite(os.path.join(temp_input_path, "1.png"), single_frame)


        # Decide the quality
        quality = str(random.randint(*opt['mpeg4_quality2']))
        preset = random.choices(opt['mpeg4_preset_mode2'], opt['mpeg4_preset_prob2'])[0]

        # Encode
        ffmpeg_encode_cmd = "ffmpeg -i " + temp_input_path + "/%d.png -vcodec libxvid -qscale:v " + quality + " -preset " + preset + " -pix_fmt yuv420p " + video_store_dir + " -loglevel 0"
        os.system(ffmpeg_encode_cmd)
        # while True:
        #     if os.path.exists(video_store_dir):
        #         # print("mpeg4 is done.")
        #         break
        #     else:
        #         # os.system(ffmpeg_encode_cmd)
        #         print("mpeg4 waiting.")


        # Decode
        ffmpeg_decode_cmd = "ffmpeg -i " + video_store_dir + " " + temp_store_path + "/%d.png -loglevel 0"
        os.system(ffmpeg_decode_cmd)
        assert(len(os.listdir(temp_store_path)) == 1)

        # Move frame to the target places
        # shutil.copy(os.path.join(temp_store_path, "1.png"), store_path)
        lq = cv2.imread(temp_store_path + "/1.png")
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        h_input, w_input = lq.shape[0:2]
        if w_input != ori_w or h_input != ori_h:
            lq = cv2.resize(lq, (ori_w, ori_h))
        # lq: numpy
        lq = lq.astype(np.float32)
        if np.max(lq) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        lq = lq / max_range
        lq = torch.from_numpy(np.transpose(lq, (2, 0, 1))).float()

        # Clean temp files
        os.remove(video_store_dir)
        shutil.rmtree(temp_input_path)
        shutil.rmtree(temp_store_path)
        return lq


    @staticmethod
    def compress_tensor(tensor_frames, idx=0):
        ''' Compress tensor input to MPEG4 and then return it (for 1st stage)
        Args:
            tensor_frame (tensor):  Tensor inputs
        Returns:
            result (tensor):        Tensor outputs (same shape as input)
        '''

        pass