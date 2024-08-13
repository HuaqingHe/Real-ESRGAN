# -*- coding: utf-8 -*-
import os


opt = {}

######################################## Setting for Degradation with Intra-Prediction ###############################################################################
opt['compression_codec2'] = ["jpeg", "webp", "avif", "mpeg2", "mpeg4", "h264", "h265"]     # Compression codec: similar to VCISR but more intense degradation settings
opt['compression_codec_prob2'] = [0.06, 0.1, 0.1, 0.12, 0.12, 0.3, 0.2] 

# Image compression setting
opt["jpeg_quality_range2"] = [20, 95]       # Harder JPEG compression setting

opt["webp_quality_range2"] = [20, 95]
opt["webp_encode_speed2"] = [0, 6]

opt["avif_quality_range2"] = [20, 95]
opt["avif_encode_speed2"] = [0, 6]          # Useless now

# Video compression I-Frame setting
opt['h264_crf_range2'] = [23, 38]
opt['h264_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
opt['h264_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

opt['h265_crf_range2'] = [28, 42]
opt['h265_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
opt['h265_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

opt['mpeg2_quality2'] = [8, 31]         #  linear scale 2-31 (the lower the higher quality)
opt['mpeg2_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
opt['mpeg2_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

opt['mpeg4_quality2'] = [8, 31]         #  should be the same as mpeg2_quality2
opt['mpeg4_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
opt['mpeg4_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

####################################################################################################################################################################

