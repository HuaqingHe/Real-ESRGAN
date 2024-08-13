import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from degradation.image_compression.jpeg import JPEG
from degradation.image_compression.webp import WEBP
from degradation.image_compression.avif import AVIF
from degradation.video_compression.h264 import H264
from degradation.video_compression.h265 import H265
from degradation.video_compression.mpeg2 import MPEG2
from degradation.video_compression.mpeg4 import MPEG4
from degradation.ESR.utils import tensor2np
import os, shutil
import ffmpeg
import cv2

@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            if self.opt['blur_flag'] is True:
                out = filter2D(self.gt_usm, self.kernel1)
            else:
                out = self.gt_usm
            # random resize
            if self.opt['resize_flag'] is True:
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            if self.opt['noise_flag'] is True:
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            # JPEG compression
            if self.opt['compression_flag'] is True:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if self.opt['blur_flag'] is True:
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
            # random resize
            if self.opt['resize_flag'] is True:
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                out = F.interpolate(
                    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            if self.opt['noise_flag'] is True:
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            # if we need compression and resize ablation studys, we can add more options.
            if self.opt['compression_flag'] is True:
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                    if self.opt['scale'] == 2:
                        out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # resize back + the final sinc filter
                    mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                    if self.opt['scale'] == 2:
                        out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
            else:
                # resize back + the final sinc filter
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                if self.opt['scale'] == 2:
                        out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)


@MODEL_REGISTRY.register()
class RealESRGANModelVideo(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModelVideo, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        # Init the compression instance
        self.jpeg_instance = JPEG()
        self.webp_instance = WEBP()
        self.avif_instance = AVIF()
        self.H264_instance = H264()
        self.H265_instance = H265()
        self.MPEG2_instance = MPEG2()
        self.MPEG4_instance = MPEG4()


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b
    @torch.no_grad()
    def add_ffmpeg_compression(self, img_lq):
        # Define the chroma subsampling factor (e.g., 2 for 4:2:0)
        chroma_subsampling_factor = 2
        # Ensure the size is a multiple of the chroma subsampling factor
        def adjust_size(width, height, factor, min_size=64):
            adjusted_width = max((width // factor) * factor, min_size)
            adjusted_height = max((height // factor) * factor, min_size)
            return adjusted_width, adjusted_height
        # img_lqs = [np.clip(img_lq * 255.0, 0, 255) for img_lq in img_lqs]
        # cv2.imwrite('001.png', cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB))
        ori_h, ori_w = img_lq.shape[0:2]
        adjusted_w, adjusted_h = adjust_size(ori_w, ori_h, chroma_subsampling_factor)
        # aovid ffmpeg input size error
        if adjusted_w != ori_w or adjusted_h != ori_h:
            img_lq = cv2.resize(img_lq, (adjusted_w, adjusted_h))
        width, height = adjusted_w, adjusted_h
        pix_fmt = 'yuv420p'
        loglevel = 'error'
        # Select a compression format with probabilities
        format = random.choices(self.opt['compression_codec2'], self.opt['compression_codec_prob2'])[0]  # All lower case
        # format = 'mpeg2'
        try:
            if format == 'h264':
                vcodec = 'libx264'
                crf = str(random.randint(*self.opt['h264_crf_range2']))
                preset = random.choices(self.opt['h264_preset_mode2'], self.opt['h264_preset_prob2'])[0]

                ffmpeg_img2video = (
                    ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                           .output('pipe:', format=format, pix_fmt=pix_fmt, crf=crf, vcodec=vcodec, preset=preset)
                           .global_args('-loglevel', loglevel)
                           .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )


            elif format == 'h265':
                format = 'hevc'         # change format 2 to hevc
                vcodec = 'libx265'
                crf = str(random.randint(*self.opt['h265_crf_range2']))
                preset = random.choices(self.opt['h265_preset_mode2'], self.opt['h265_preset_prob2'])[0]
                x265_params = 'log-level=error'

                ffmpeg_img2video = (
                    ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                           .output('pipe:', format=format, pix_fmt=pix_fmt, crf=crf, vcodec=vcodec, preset=preset)
                           .global_args('-loglevel', loglevel).global_args('-x265-params', x265_params)
                           .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )


            elif format == 'mpeg2':
                #  img2video format is mpeg2video and video2img format is mpeg2
                format = 'mpeg'
                vcodec = 'mpeg2video'
                quality = str(random.randint(*self.opt['mpeg2_quality2']))
                preset = random.choices(self.opt['mpeg2_preset_mode2'], self.opt['mpeg2_preset_prob2'])[0]

                ffmpeg_img2video = (
                    ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                           .output('pipe:', format=format, pix_fmt=pix_fmt, **{'q:v': quality}, vcodec=vcodec, preset=preset)
                           .global_args('-loglevel', loglevel)
                           .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )


            elif format == 'mpeg4':
                format = 'mpeg'
                vcodec = 'libxvid'
                quality = str(random.randint(*self.opt['mpeg4_quality2']))
                preset = random.choices(self.opt['mpeg4_preset_mode2'], self.opt['mpeg4_preset_prob2'])[0]

                ffmpeg_img2video = (
                    ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                           .output('pipe:', format=format, pix_fmt=pix_fmt, **{'q:v': quality}, vcodec=vcodec, preset=preset)
                           .global_args('-loglevel', loglevel)
                           .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )

            ffmpeg_video2img = (
                ffmpeg.input('pipe:', format=format)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .global_args('-hide_banner').global_args('-loglevel', loglevel)
                    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))

            # read a sequence of images
            ffmpeg_img2video.stdin.write(img_lq.astype(np.uint8).tobytes())
            ffmpeg_img2video.stdin.close()

            video_bytes = ffmpeg_img2video.stdout.read()
            ffmpeg_img2video.wait()
            ffmpeg_img2video.stdout.close()
            # print stderr for debug
            # stderr_output = ffmpeg_img2video.stderr.read().decode('utf-8')
            # print("FFmpeg stderr output for img2video:", stderr_output)
            # # close stdout and stderr
            # ffmpeg_img2video.stdout.close()
            # ffmpeg_img2video.stderr.close()

            # ffmpeg: video to images
            ffmpeg_video2img.stdin.write(video_bytes)
            ffmpeg_video2img.stdin.flush()  # Ensure all data is written
            ffmpeg_video2img.stdin.close()

            # print stderr for debug
            # stderr_output = ffmpeg_video2img.stderr.read().decode('utf-8')
            # ffmpeg_video2img.stderr.close()
            # print("FFmpeg stderr output for video2img:", stderr_output)


            while True:
                in_bytes = ffmpeg_video2img.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                img_lq = (np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))
                # print('in_bytes:', np.frombuffer(in_bytes, np.uint8))
                # print('in_frame.shape:', in_frame.shape)
                # resize back
                # cv2.imwrite('002.png', cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB))
                # in_frame = in_frame.astype(np.float32) / 255.


            ffmpeg_video2img.wait()
            ffmpeg_video2img.stdout.close()
            # assert len(img_lqs_ffmpeg) == self.num_frame, 'Wrong length'
        except AssertionError as error:
            print(f'ffmpeg assertion error: {error}')
        except Exception as error:
            print(f'ffmpeg exception error: {error}')
        else:
            if adjusted_w != ori_w or adjusted_h != ori_h:
                    img_lq = cv2.resize(img_lq, (ori_w, ori_h))
            img_lq = img_lq.astype(np.float32)
            if np.max(img_lq) > 256:  # 16-bit image
                max_range = 65535
                print('\tInput is a 16-bit image')
            else:
                max_range = 255
            img_lq = img_lq / max_range
            img_lq = torch.from_numpy(np.transpose(img_lq, (2, 0, 1))).float()
            return img_lq
        
    @torch.no_grad()
    def compression(self, np_frame, idx=0, tmp_path='tmp'):
        # Choose an image compression codec (All degradation batch use the same codec)
        compression_codec = random.choices(self.opt['compression_codec2'], self.opt['compression_codec_prob2'])[0]  # All lower case
        if compression_codec == 'jpeg':
            self.jpeg_instance.compress_and_store(np_frame)
        elif compression_codec == "webp":
            try:
                self.webp_instance.compress_and_store(np_frame)
            except Exception:
                print("There appears to be exception in webp again!")
                self.webp_instance.compress_and_store(np_frame)
        elif compression_codec == "avif":
            self.avif_instance.compress_and_store(np_frame)
        elif compression_codec == "h264":
            np_frame = self.H264_instance.compress_and_store(self.opt, np_frame, idx, tmp_path)
        elif compression_codec == "h265":
            np_frame = self.H265_instance.compress_and_store(self.opt, np_frame, idx, tmp_path)

        elif compression_codec == "mpeg2":
            np_frame = self.MPEG2_instance.compress_and_store(self.opt, np_frame, idx, tmp_path)

        elif compression_codec == "mpeg4":
            np_frame = self.MPEG4_instance.compress_and_store(self.opt, np_frame, idx, tmp_path)

        else:
            raise NotImplementedError("This compression codec is not supported! Please check the implementation!")
        return np_frame

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            if self.opt['blur_flag'] is True:
                out = filter2D(self.gt_usm, self.kernel1)
            else:
                out = self.gt_usm
            # random resize
            if self.opt['resize_flag'] is True:
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            if self.opt['noise_flag'] is True:
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            # JPEG compression
            if self.opt['JPEG_flag'] is True:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if self.opt['blur_flag'] is True:
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)
            # random resize
            if self.opt['resize_flag'] is True:
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            if self.opt['noise_flag'] is True:
                final_noise_flag = np.random.uniform()
                if 'FinalNoise_flag' in self.opt.keys() and self.opt['FinalNoise_flag'] is True:
                    if final_noise_flag >= self.opt['final_noise_prob']:     # add noise before compression
                        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                            out = random_add_gaussian_noise_pt(
                                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                        else:
                            out = random_add_poisson_noise_pt(
                                out,
                                scale_range=self.opt['poisson_scale_range2'],
                                gray_prob=gray_noise_prob,
                                clip=True,
                                rounds=False)
                else:
                    gray_noise_prob = self.opt['gray_noise_prob2']
                    if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                        out = random_add_gaussian_noise_pt(
                            out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    else:
                        out = random_add_poisson_noise_pt(
                            out,
                            scale_range=self.opt['poisson_scale_range2'],
                            gray_prob=gray_noise_prob,
                            clip=True,
                            rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            # if we need compression and resize ablation studys, we can add more options.
            if self.opt['compression_flag'] is True:
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                    if self.opt['scale'] == 2:
                        out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
                    # # JPEG compression
                    # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    # out = torch.clamp(out, 0, 1)
                    # out = self.jpeger(out, quality=jpeg_p)
                    ### JPEG compression -> Video compression
                    out = torch.clamp(out, 0, 1)    # 24 x 3 x 400 x 400 change it to 3 x 400 x 400. RGB
                    # Initialize a list to store processed frames
                    processed_frames = []
                    # cuda_number = torch.cuda.current_device()
                    # Loop through each batch
                    for i in range(out.size(0)):
                        # Extract the i-th frame and convert it to numpy array
                        np_frame = tensor2np(out[i])  # 3 x 400 x 400
                        # idx = cuda_number * out.size(0) + i
                        # np_frame = self.compression(np_frame, idx, self.opt['tmp_path'])
                        np_frame = self.add_ffmpeg_compression(np_frame)
                        # Convert the processed numpy array back to tensor and append to the list
                        processed_frames.append(np_frame)
                    # Stack the processed frames back into a single tensor
                    out = torch.stack(processed_frames).to(self.device)
                else:
                    # # JPEG compression
                    # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    # out = torch.clamp(out, 0, 1)
                    # out = self.jpeger(out, quality=jpeg_p)
                    ### JPEG compression -> Video compression
                    out = torch.clamp(out, 0, 1)    # 24 x 3 x 400 x 400 change it to 3 x 400 x 400. RGB
                    # Initialize a list to store processed frames
                    processed_frames = []
                    # cuda_number = torch.cuda.current_device()
                    # Loop through each batch
                    for i in range(out.size(0)):
                        # Extract the i-th frame and convert it to numpy array
                        np_frame = tensor2np(out[i])  # 3 x 400 x 400
                        # idx = cuda_number * out.size(0) + i

                        # np_frame = self.compression(np_frame, idx, self.opt['tmp_path'])
                        np_frame = self.add_ffmpeg_compression(np_frame)
                        # Convert the processed numpy array back to tensor and append to the list
                        processed_frames.append(np_frame)

                    # Stack the processed frames back into a single tensor
                    out = torch.stack(processed_frames).to(self.device)

                    # resize back + the final sinc filter
                    mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                    if self.opt['scale'] == 2:
                        out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out = filter2D(out, self.sinc_kernel)
            else:
                # resize back + the final sinc filter
                mode = random.choice(['nearest', 'bilinear', 'bicubic'])
                if self.opt['scale'] == 2:
                    out = F.interpolate(out, size=(ori_h // 4, ori_w // 4), mode=mode)
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
            if 'FinalBlur_flag' in self.opt.keys() and self.opt['FinalBlur_flag'] is True:
                if np.random.uniform() < self.opt['final_blur_prob']:
                    out = filter2D(out, self.kernel2)
            if 'FinalNoise_flag' in self.opt.keys() and self.opt['FinalNoise_flag'] is True:
                if final_noise_flag < self.opt['final_noise_prob']:     # add noise before compression
                    if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                        out = random_add_gaussian_noise_pt(
                            out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    else:
                        out = random_add_poisson_noise_pt(
                            out,
                            scale_range=self.opt['poisson_scale_range2'],
                            gray_prob=gray_noise_prob,
                            clip=True,
                            rounds=False)
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRGANModelVideo, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
