import argparse
import math
import torch
from torchvision import utils
from IPython.display import clear_output
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../")
from stylegan_models import StyledGenerator
import os 
import torch.nn.functional as F
from tqdm import tqdm
from torch.quantization.observer import *
from torch.quantization.fake_quantize import *
from torch.quantization.qconfig import *
from torch.quantization import prepare_qat
from torch.quantization import prepare
import torch.nn.qat as nnq
from utils.quant import MovingAverageQuantileObserver, MovingAveragePerChannelQuantileObserver
from utils.learnable_fake_quantize import _LearnableFakeQuantize, enable_param_learning, \
                                        enable_static_estimate, enable_static_observation, enable_fixed_estimate

from IPython.display import clear_output
import numpy as np
import glob
import torchvision.transforms as transforms
from datasets.datasets import PairedImageDataset
from torch.utils.data import DataLoader
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from torch.autograd import Variable
from utils.fid_score import calculate_fid_given_paths

incl = torch.quantization.quantization_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST
incl.add(nn.LeakyReLU)
torch.quantization.quantization_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST = incl
torch.quantization.quantization_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST

from utils.adaround import AdaRound, enable_sigmoid_quant, LossFunction, weighted_mse, enable_hard_sigmoid



class FixedMovingAveragePerChannelQuantileObserver(MovingAveragePerChannelQuantileObserver):
    def __init__(self, averaging_constant=0.01, ch_axis=0, dtype=torch.quint8,
                 qscheme=torch.per_channel_symmetric, reduce_range=False,
                 quant_obs_min=None, quant_obs_max=None, q_min=0.0, q_max=1.0):
        super(FixedMovingAveragePerChannelQuantileObserver, self).__init__(averaging_constant=averaging_constant,
            ch_axis=ch_axis, dtype=dtype, qscheme=qscheme, q_max=q_max, q_min=q_min,
            reduce_range=reduce_range, quant_min=quant_obs_min, quant_max=quant_obs_max)
        
class FixedMovingAverageQuantileObserver(MovingAverageQuantileObserver):
    def __init__(self, averaging_constant=0.01, dtype=torch.quint8,
                 qscheme=torch.per_channel_symmetric, reduce_range=False,
                 quant_obs_min=None, quant_obs_max=None, q_min=0.0, q_max=1.0):
        super(FixedMovingAverageQuantileObserver, self).__init__(averaging_constant=averaging_constant,
            dtype=dtype, qscheme=qscheme, q_min=q_min, q_max=q_max,
            reduce_range=reduce_range, quant_obs_min=quant_obs_min, quant_obs_max=quant_obs_max)

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, n_sample, device="cuda",  step=5, mean_style=None, seed=23):
    torch.manual_seed(seed)
    generator.to(device)
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image


def calibrate_model(model, num_forwards=100, batch_size=1, cpu_inp=False):
    model.apply(enable_static_estimate)
    for i in tqdm(range(num_forwards), desc='Calibrating...'):
        sample(model, batch_size, seed=i, device="cpu" if cpu_inp else "cuda")
    return model

def get_q_config(q_params):

    if q_params.get("per_channel", True):
        qscheme_w = torch.per_channel_symmetric
        obs_w = FixedMovingAveragePerChannelQuantileObserver
    else:
        qscheme_w = torch.per_tensor_symmetric
        obs_w = FixedMovingAverageQuantileObserver        
    
    assert not (q_params.get("lsq_weight", False) and q_params.get("adaround", False)), "Choose LSQ or Adaround, not both"
    
    weight_kwargs = {"observer":obs_w,
                     "quant_min":-(2 **(q_params.get('bits_w', 8)-1)),
                     "quant_max":(2 ** (q_params.get('bits_w', 8)-1)) - 1,
                     "dtype":torch.qint8,
                     "qscheme":qscheme_w,
                     "quant_obs_min":-(2 ** (q_params.get('bits_w', 8)-1)),
                     "quant_obs_max":(2 ** (q_params.get('bits_w', 8)-1)) - 1,
                     "q_min":q_params.get("q_min_w", 0.0),
                     "q_max":q_params.get("q_max_w", 1.0)}
                     
    if q_params.get("lsq_weight", False):
        weight_kwargs["weight"] = True
        weight_kwargs["use_grad_scaling"] = True
        weight_quant = _LearnableFakeQuantize.with_args(**weight_kwargs)
    elif q_params.get("adaround", False):
        weight_quant = AdaRound.with_args(**weight_kwargs)                                          
    else:
        weight_quant = FakeQuantize.with_args(**weight_kwargs)
            
    if q_params.get("lsq_act"):
        fq_act = _LearnableFakeQuantize
    else:
        fq_act = FakeQuantize

    if q_params.get('hist_act'):
        act_quant = fq_act.with_args(observer=HistogramObserver,
                                    dtype=torch.quint8,
                                    reduce_range=q_params.get('reduce_hist_act', False))
    else:
        act_quant = fq_act.with_args(observer=MovingAverageQuantileObserver,
                                    dtype=torch.quint8,
                                    q_min=q_params['q_min'], q_max=q_params['q_max'])

    qconfig = QConfig(activation=act_quant,
                      weight=weight_quant)
    return qconfig


def get_static_quant_model(model_path, q_params=None, numits_calib=500, get_fp=False, disable_observer=True):
    
    model = StyledGenerator(512)
    model.load_state_dict(torch.load(model_path))

    if get_fp:
        return model.eval()
    
    model.qconfig = get_q_config(q_params)
    # model.eval()
    model.cpu()

    for m in model.modules():
        if (type(m) == nn.Linear) or (type(m) == nn.Conv2d):
            with torch.no_grad():
                m.weight = nn.Parameter(m.weight_orig.data).to(m.weight_orig)
                del m.weight_orig
                
    prepare_qat(model, inplace=True)

    for m in model.modules():
        if (type(m) == nnq.Linear) or (type(m) == nnq.Conv2d):
            with torch.no_grad():
                m.weight_orig = nn.Parameter(m.weight.data).to(m.weight)
                del m.weight



    if q_params.get('hist_act'):
        calibrate_model(model, numits_calib, cpu_inp=True)
        model = model.cpu()
    else:
        model = model.cuda()
        calibrate_model(model, numits_calib)
        model = model.cuda()

    if disable_observer:
        model.apply(torch.quantization.disable_observer)
        model.apply(enable_fixed_estimate)



    #netG = torch.quantization.convert(netG.cpu().eval(), inplace=False)
    return model.eval()


def compute_fid(netG, data_dir, reference_data,
                cpu_inference=False, data_size=50000, delete_cache=False):

    original_data_path = reference_data + "/distil_pics/"

    if delete_cache:
        for file_img in glob.glob(data_dir + '/*.png'):
            os.remove(file_img)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if len(glob.glob(data_dir + '/*')) < 50000:
        print("Here is no generated data, so I generate it using provided model")

        b_size = 50

        eval_dataloader = DataLoader(
            PairedImageDataset(reference_data),
            batch_size=b_size, shuffle=False, num_workers=4, drop_last=False)

        input_eval_source = torch.cuda.FloatTensor(b_size, 512)
        netG.eval()
        for i_eval_img, eval_batch in tqdm(enumerate(eval_dataloader)):
            input_img = Variable(input_eval_source.copy_(eval_batch['input']))
            with torch.no_grad():

                if cpu_inference:
                    input_img = input_img.cpu()
                output_img = netG(input_img)

            for i_img_from_batch in range(b_size):
                img_np = output_img[i_img_from_batch:(i_img_from_batch+1)].detach().cpu().numpy()

                img_np = np.moveaxis(img_np, 1, -1)
                img_np = np.clip((img_np + 1) / 2, 0, 1)  # (-1,1) -> (0,1)

                imsave(os.path.join(data_dir, '%s.png' % (i_eval_img * b_size + i_img_from_batch)), img_as_ubyte(img_np[0]))

                if i_eval_img + 1 == data_size:
                    break
    else:
        pass
        #print(f"I found {len(glob.glob(data_dir + '/*.png'))} pictures in the folder")
    paths = [data_dir, original_data_path]

    fid = calculate_fid_given_paths(paths, 32, True, 2048, delete_cache)
    return fid


def get_fid(model, data_dir, reference_data, qfid=True, use_cache=True):
    
    try:
        if use_cache:
            fid_sc = np.load(data_dir + f'/fid_full_{qfid}.npy')[0]
        else:
            raise Exception
    except:
        if qfid:
            fid_sc = compute_fid(model,
                      cpu_inference=False,
                      delete_cache=not use_cache, reference_data=reference_data,
                      data_dir=data_dir, data_size=50000)
        else:
            assert len(glob.glob(data_dir + '/*')) >= 50000, "Please run compute qfid at first"
            fid_sc = calculate_fid_given_paths([data_dir, reference_data], 32, True, 2048, not use_cache)
        
        np.save(data_dir + f'/fid_full_{qfid}.npy', np.array([fid_sc]))
        
    return fid_sc

def get_fid_from_qparams(q_params, data_dir, model_path, reference_data, qfid=True, use_cache=True):
    try:
        if use_cache:
            fid_sc = np.load(data_dir + f'/fid_full_{qfid}.npy')[0]
        else:
            raise Exception
    except: 
        if qfid:
            model = get_static_quant_model(model_path, q_params, numits_calib=500).cuda()
        else:
            model = get_static_quant_model(model_path, q_params, numits_calib=1).cuda()
        fid_sc = get_fid(model, data_dir, reference_data, qfid, use_cache)
    return fid_sc

def transform_style(generator):
    style = []
    style.append(nn.Sequential(generator.quant_input, generator.style[0], generator.style[1],
                               generator.style[2]))
    for i in range(3, len(generator.style), 2):
        style.append(nn.Sequential(generator.style[i], generator.style[i+1]))
    style = nn.Sequential(*style)
    generator.style = style
    generator.quant_input = nn.Identity()
    return 

def get_pretrained_model(path, fp_path, q_params, get_fp=False):
    
    model = get_static_quant_model(fp_path,
                                   q_params=q_params,
                                   numits_calib=1,
                                   get_fp=get_fp).cpu()
    if get_fp:
        return model
    
    if q_params["lsq_weight"] or q_params["adaround"]:
        transform_style(model)
        
    model.apply(enable_sigmoid_quant)
    sample(model, 1, "cpu")
    model.apply(enable_fixed_estimate)
    model.apply(enable_hard_sigmoid)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.apply(torch.quantization.disable_observer)
    model.apply(enable_fixed_estimate)
    
    return model