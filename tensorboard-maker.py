import re
from torch.utils.tensorboard import SummaryWriter

# 步骤1: 解析日志文件
log_file_path = '/root/picasso/HuaqingHe/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k/train_finetune_RealESRGANx4plus_400k_20240716_154615.log'

# 存储解析的数据
data = []

with open(log_file_path, 'r') as file:
    for line in file:
        if 'epoch' in line and 'iter' in line and 'lr' in line and 'eta' in line and 'time' in line and 'l_g_pix' in line and 'l_g_percep' in line and 'l_g_gan' in line and 'l_d_real' in line and 'out_d_real' in line and 'l_d_fake' in line and 'out_d_fake' in line:
            # 提取这行里的数据
            epoch = int(line.split('epoch:')[1].split(',')[0].strip())
            iter = int(line.split('iter:')[1].split(',')[0].strip())
            lr = float(line.split('lr:(')[1].split(',')[0].strip())
            l_g_pix = float(line.split('l_g_pix:')[1].split('l_g_percep')[0].strip())
            l_g_percep = float(line.split('l_g_percep:')[1].split('l_g_gan')[0].strip())
            l_g_gan = float(line.split('l_g_gan:')[1].split('l_d_real')[0].strip())
            l_d_real = float(line.split('l_d_real:')[1].split('out_d_real')[0].strip())
            out_d_real = float(line.split('out_d_real:')[1].split('l_d_fake')[0].strip())
            l_d_fake = float(line.split('l_d_fake:')[1].split('out_d_fake')[0].strip())
            out_d_fake = float(line.split('out_d_fake:')[1].strip())
            data.append((epoch, iter, lr, l_g_pix, l_g_percep, l_g_gan, l_d_real, out_d_real, l_d_fake, out_d_fake))


# 步骤2: 使用TensorBoard记录数据

# 生成sumary文件的路径是日志文件的上层路径
sumary_path = log_file_path.split('.log')[0]
writer = SummaryWriter(sumary_path)

for i, (epoch, iter, lr, l_g_pix, l_g_percep, l_g_gan, l_d_real, out_d_real, l_d_fake, out_d_fake) in enumerate(data):
    writer.add_scalar('iter', iter, epoch)
    writer.add_scalar('lr', lr, epoch)
    writer.add_scalar('l_g_pix', l_g_pix, epoch)
    writer.add_scalar('l_g_percep', l_g_percep, epoch)
    writer.add_scalar('l_g_gan', l_g_gan, epoch)
    writer.add_scalar('l_d_real', l_d_real, epoch)
    writer.add_scalar('out_d_real', out_d_real, epoch)
    writer.add_scalar('l_d_fake', l_d_fake, epoch)
    writer.add_scalar('out_d_fake', out_d_fake, epoch)

writer.close()