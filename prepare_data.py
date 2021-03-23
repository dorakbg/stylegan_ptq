import torch
import argparse
import math
from torchvision import utils
import sys
sys.path.insert(0, "../")
from stylegan_models import StyledGenerator
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import glob
import shutil

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run inference')
parser.add_argument('--weights', type=str, default="./weights.pth", help='Path to full-precision StyleGAN weights')
parser.add_argument('--ffhq_path', type=str, required=True, help='Path to ffhq data 128x128, please download the data beforehand from https://github.com/NVlabs/ffhq-dataset, (python download_ffhq.py -h), and provide path to thumbnails128x128 directory in this argument, example: "/data/datasets/ffhq-dataset/thumbnails128x128"')
parser.add_argument('--save_data_dir', type=str, required=True, help='Provide path to directory where to save the generated data, the script creates two directories "fp_data" and "real_images", where it places images needed for FID and qFID computation')

args = parser.parse_args()

device = args.device


# generate full-precision data

output_dir = os.path.join(args.save_data_dir, "fp_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_dir + '/distil_noise')
    os.makedirs(output_dir + '/distil_pics')

generator = StyledGenerator(512).to(device)
generator.load_state_dict(torch.load(args.weights)) 
generator.eval()

torch.cuda.manual_seed(0)
torch.manual_seed(0)

for i in tqdm(range(50000)):
    z_sample = torch.randn(1, 512)
    with torch.no_grad():
        image = denorm(generator(tensor2var(z_sample)))
    save_image(image, f'{output_dir}/distil_pics/{i}.png')
    torch.save(z_sample, f'{output_dir}/distil_noise/{i}.pth')


# prepare real data for FID computation

output_dir = os.path.join(args.save_data_dir, "real_images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = glob.glob(f"{args.ffhq_path}/*/*")
    
for i in tqdm(range(50000)):
    shutil.copy(files[i], output_dir + f"/{i}.png")
    

