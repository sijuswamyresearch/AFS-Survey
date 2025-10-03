# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import math
import warnings
import time
import copy
import traceback
from collections import defaultdict
from functools import partial
import glob
import argparse # Added
import pickle   # Added

from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. Running without HPO. Using default LR.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS library not found. LPIPS metric will not be calculated.")

THOP_AVAILABLE = False # Disabled

# --- Custom Activation Functions ---
class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.tanh(F.softplus(x))
class Swish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.sigmoid(x)
class ESwish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__(); self.beta = nn.Parameter(torch.tensor(float(beta)))
    def forward(self, x): return self.beta * x * torch.sigmoid(x)
class Aria(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__(); self.alpha = nn.Parameter(torch.tensor(float(alpha))); self.beta = nn.Parameter(torch.tensor(float(beta)))
    def forward(self, x):
        x = torch.clamp(x, -20, 20); exp_term = torch.exp(-self.alpha * x); sin_term = torch.sin(self.beta * x)
        return exp_term * sin_term
class GCU(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.cos(torch.clamp(x, -math.pi, math.pi))
class Snake(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__(); self.alpha = nn.Parameter(torch.tensor(float(alpha)))
    def forward(self, x):
        alpha_safe = torch.where(self.alpha == 0, torch.tensor(1e-7, device=self.alpha.device, dtype=self.alpha.dtype), self.alpha)
        return x + (torch.sin(alpha_safe * x) ** 2 / alpha_safe)
class FReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0: raise ValueError(f"FReLU requires positive int channels, got {channels}")
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1); nn.init.constant_(self.bn.bias, 0)
    def forward(self, x):
        if x.dim() != 4: raise ValueError(f"FReLU expects 4D input (NCHW), got {x.dim()}D")
        if x.size(1) != self.channels: raise ValueError(f"Input channels {x.size(1)} != FReLU channels {self.channels}")
        spatial_cond = self.bn(self.conv(x)); out = torch.max(x, spatial_cond)
        return out
class FAATanh(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=1.0, gamma_init=1.0, delta_init=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init))); self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))); self.delta = nn.Parameter(torch.tensor(float(delta_init)))
    def forward(self, x):
        tanh_term = torch.tanh(self.beta * x); gate_input = self.gamma * torch.abs(x) - self.delta
        gate = torch.sigmoid(gate_input); output = x + self.alpha * tanh_term * (1.0 - gate)
        return output
class SAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0: raise ValueError(f"SAG positive int channels, got {channels}")
        self.channels = channels
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.spatial_bn.weight, 1); nn.init.constant_(self.spatial_bn.bias, 0)
    def forward(self, x):
        if x.dim() != 4: raise ValueError("SAG expects 4D input (NCHW)")
        if x.size(1) != self.channels: raise ValueError(f"Input channels {x.size(1)} != SAG channels {self.channels}")
        spatial_cond = self.spatial_bn(self.spatial_conv(x)); gate = torch.sigmoid(spatial_cond); output = x * gate
        return output
class CST_SAGA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0: raise ValueError(f"CST-SAGA positive int channels, got {channels}")
        self.channels = channels
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels); self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.spatial_bn.weight, 1); nn.init.constant_(self.spatial_bn.bias, 0)
    def forward(self, x):
        if x.dim() != 4: raise ValueError("expects 4D input (NCHW)")
        if x.size(1) != self.channels: raise ValueError(f"Input channels {x.size(1)} != CST-SAGA channels {self.channels}")
        spatial_cond = self.spatial_bn(self.spatial_conv(x)); boost = F.relu(spatial_cond - x); output = x + self.alpha * boost
        return output

# --- Configuration ---
MODEL_CHOICE = "EDSR" # Options: "UNET", "EDSR"
print(f"INFO: Using base model: {MODEL_CHOICE}")

DATASET_NAME = "CT_dataset" # Options: "CT_dataset", "BC_dataset","BR35H_dataset"
TASK_NAME = "data_dblur"
DEGRADED_FOLDER_NAME = "blur"
DATASET_PARENT_DIR = "/dist_home/siju/AFS/"

if DATASET_NAME == "HAM10000": N_CHANNELS = 3 # for generalization
else: N_CHANNELS = 1
print(f"INFO: Model I/O channels: {N_CHANNELS} (based on DATASET_NAME='{DATASET_NAME}')")

EXPERIMENT_FOLDER_NAME = f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_activation_analysis"
DATASET_ROOT = os.path.join(DATASET_PARENT_DIR, DATASET_NAME)
MODEL_DIR = os.path.join("models", EXPERIMENT_FOLDER_NAME)
RESULT_DIR = os.path.join("results", EXPERIMENT_FOLDER_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "spatial_analysis"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "fft_analysis"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "comparison_plots"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "loss_curves"), exist_ok=True)

BATCH_SIZE = 8 #  can change batch size
EPOCHS = 100 # Reduced for quick testing
TARGET_SIZE = (256, 256)
NUM_RUNS = 3 # Reduced for quick testing
GLOBAL_WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15

HPO_ENABLED = OPTUNA_AVAILABLE and False # HPO often takes significant time per activation
HPO_N_TRIALS = 10
HPO_N_EPOCHS = 3
DEFAULT_LR = 1e-4
LR_RANGE = [1e-5, 5e-4]

EDSR_N_RESBLOCKS = 16
EDSR_N_FEATS = 64
EDSR_RES_SCALE = 0.1

PARAMETRIC_ACTIVATIONS_TYPES = (ESwish, Aria, Snake, FReLU, FAATanh, SAG, CST_SAGA, nn.PReLU)

# Store all activations here, this will be split if running in parts
ALL_ACTIVATIONS_CONFIG = {
    "ReLU": {"fn": nn.ReLU()}, "LeakyReLU": {"fn": nn.LeakyReLU(0.1)}, "PReLU": {"fn": nn.PReLU()},"GELU": {"fn": nn.GELU()},
    "Sigmoid": {"fn": nn.Sigmoid()}, "Tanh": {"fn": nn.Tanh()}, "ELU": {"fn": nn.ELU(alpha=1.0)},
    "SiLU": {"fn": nn.SiLU()}, "Mish": {"fn": Mish()}, "Swish": {"fn": Swish()},
    "ESwish": {"fn": ESwish(beta=1.25)}, "Aria": {"fn": Aria(alpha=0.5, beta=1.0)}, "GCU": {"fn": GCU()},
    "Snake": {"fn": Snake(alpha=0.5)},
    "FReLU": {"fn": FReLU(channels=EDSR_N_FEATS if MODEL_CHOICE=="EDSR" else 64)},
    "FAATanh": {"fn": FAATanh()}

}
# --- Medical Image Dataset ---
class MedicalImageRestorationDataset(Dataset):
    def __init__(self, root_dir, split='train', task_folder='data_dblur', degraded_folder='blur',
                 transform=None, target_size=None, input_channels=3):
        self.task_split_path = os.path.join(root_dir, task_folder, split)
        self.sharp_dir = os.path.join(self.task_split_path, 'sharp')
        self.degraded_dir = os.path.join(self.task_split_path, degraded_folder)
        self.transform = transform
        self.target_size = target_size
        self.input_channels = input_channels

        if not os.path.isdir(self.sharp_dir): raise FileNotFoundError(f"Sharp dir not found: {self.sharp_dir}")
        if not os.path.isdir(self.degraded_dir): raise FileNotFoundError(f"Degraded dir not found: {self.degraded_dir}")

        supported_extensions = ['*.pt']
        self.sharp_files = []
        for ext in supported_extensions:
            self.sharp_files.extend(sorted(glob.glob(os.path.join(self.sharp_dir, ext))))

        if not self.sharp_files: raise FileNotFoundError(f"No sharp tensors found in {self.sharp_dir} with extension .pt")

        self.image_pairs = []
        self.image_ids = []
        missing_counter = 0
        for sharp_path in self.sharp_files:
            filename = os.path.basename(sharp_path)
            degraded_path = os.path.join(self.degraded_dir, filename)
            if os.path.isfile(degraded_path):
                self.image_pairs.append((sharp_path, degraded_path))
                self.image_ids.append(filename)
            else:
                missing_counter += 1
        if missing_counter > 0: print(f"Warning: Found {missing_counter} sharp tensors without corresponding degraded tensor in {self.degraded_dir}.")
        if not self.image_pairs: raise FileNotFoundError(f"No valid tensor pairs found between {self.sharp_dir} and {self.degraded_dir}")
        # print(f"Found {len(self.image_pairs)} tensor pairs for task '{task_folder}', split '{split}'.") # Moved to main

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        sharp_path, degraded_path = self.image_pairs[idx]
        image_id = self.image_ids[idx]

        current_h, current_w = (256, 256)
        if self.target_size:
            current_h, current_w = self.target_size

        try:
            try:
                sharp_tensor = torch.load(sharp_path, weights_only=False).float()
                degraded_tensor = torch.load(degraded_path, weights_only=False).float()
            except TypeError: # Fallback for older PyTorch versions
                sharp_tensor = torch.load(sharp_path).float()
                degraded_tensor = torch.load(degraded_path).float()


            if not isinstance(sharp_tensor, torch.Tensor) or not isinstance(degraded_tensor, torch.Tensor):
                raise TypeError("Loaded file is not a PyTorch tensor.")

            if sharp_tensor.dim() == 3 and sharp_tensor.shape[2] == self.input_channels: sharp_tensor = sharp_tensor.permute(2, 0, 1)
            elif sharp_tensor.dim() == 2 and self.input_channels == 1: sharp_tensor = sharp_tensor.unsqueeze(0)
            if degraded_tensor.dim() == 3 and degraded_tensor.shape[2] == self.input_channels: degraded_tensor = degraded_tensor.permute(2, 0, 1)
            elif degraded_tensor.dim() == 2 and self.input_channels == 1: degraded_tensor = degraded_tensor.unsqueeze(0)

            if sharp_tensor.dim() != 3 or sharp_tensor.shape[0] != self.input_channels: raise ValueError(f"Sharp tensor shape mismatch. Expected C={self.input_channels}, Got {sharp_tensor.shape}")
            if degraded_tensor.dim() != 3 or degraded_tensor.shape[0] != self.input_channels: raise ValueError(f"Degraded tensor shape mismatch. Expected C={self.input_channels}, Got {degraded_tensor.shape}")

            sharp_tensor = torch.clamp(sharp_tensor, 0.0, 1.0); degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)

            if not self.target_size:
                current_h, current_w = sharp_tensor.shape[1], sharp_tensor.shape[2]

        except Exception as e_load:
            # print(f"Error loading/processing tensor (idx {idx}, ID: {image_id}): {e_load}") # Can be verbose
            return torch.zeros((self.input_channels, current_h, current_w)), torch.zeros((self.input_channels, current_h, current_w)), f"error_load_{image_id}"

        if self.target_size:
            resize_transform = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            try:
                sharp_tensor = resize_transform(sharp_tensor)
                degraded_tensor = resize_transform(degraded_tensor)
            except Exception as e_resize:
                # print(f"Error resizing tensor (idx {idx}, ID: {image_id}): {e_resize}")
                return torch.zeros((self.input_channels, current_h, current_w)), torch.zeros((self.input_channels, current_h, current_w)), f"error_resize_{image_id}"

        if self.transform:
            apply_hflip_pair = False
            if any(isinstance(t, transforms.RandomHorizontalFlip) for t in (self.transform.transforms if isinstance(self.transform, transforms.Compose) else [self.transform])):
                if random.random() > 0.5:
                    apply_hflip_pair = True

            temp_transform_list = []
            if isinstance(self.transform, transforms.Compose):
                temp_transform_list = [t for t in self.transform.transforms if not isinstance(t, transforms.RandomHorizontalFlip)]
            elif not isinstance(self.transform, transforms.RandomHorizontalFlip):
                temp_transform_list = [self.transform]

            if apply_hflip_pair:
                sharp_tensor = transforms.functional.hflip(sharp_tensor)
                degraded_tensor = transforms.functional.hflip(degraded_tensor)

            if temp_transform_list:
                final_composed_transform = transforms.Compose(temp_transform_list)
                try:
                    sharp_tensor = final_composed_transform(sharp_tensor)
                    degraded_tensor = final_composed_transform(degraded_tensor)
                except Exception as e_transform:
                    # print(f"Error applying transforms (idx {idx}, ID: {image_id}): {e_transform}")
                    return torch.zeros((self.input_channels, current_h, current_w)), torch.zeros((self.input_channels, current_h, current_w)), f"error_transform_{image_id}"
        return degraded_tensor, sharp_tensor, image_id

# --- U-Net architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation_fn=nn.ReLU()):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        if not isinstance(activation_fn, nn.Module): raise TypeError("activation_fn must be an nn.Module instance")
        def _get_act_instance(act_template, channels_for_act):
            if isinstance(act_template, FReLU): return FReLU(channels=channels_for_act)
            elif isinstance(act_template, SAG): return SAG(channels=channels_for_act)
            elif isinstance(act_template, CST_SAGA): return CST_SAGA(channels=channels_for_act)
            elif isinstance(act_template, nn.PReLU): return nn.PReLU(num_parameters=1) # Standard PReLU
            elif isinstance(act_template, PARAMETRIC_ACTIVATIONS_TYPES): return copy.deepcopy(act_template)
            else: return copy.deepcopy(act_template)
        self.activation1 = _get_act_instance(activation_fn, mid_channels)
        self.activation2 = _get_act_instance(activation_fn, out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), self.activation1,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), self.activation2)
    def forward(self, x): return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, activation_fn=activation_fn))
    def forward(self, x): return self.maxpool_conv(x)
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, activation_fn, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1) # Reduce channels after upsample
            conv_in_channels = skip_channels + out_channels # Adjusted for reduction
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels , kernel_size=2, stride=2) # ConvT output is out_channels
            conv_in_channels = skip_channels + out_channels # Adjusted

        self.conv = DoubleConv(conv_in_channels, out_channels, activation_fn=activation_fn)

    def forward(self, x1, x2): # x1 is from previous layer (to be upsampled), x2 is skip connection
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_reduce(x1) # Reduce channels of x1 before cat

        # Pad x1 to match x2's spatial dimensions if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=nn.ReLU(), bilinear=True): # Use global N_CHANNELS
        super(UNet, self).__init__()
        if n_channels <= 0 or n_classes <= 0: raise ValueError("n_channels/n_classes must be positive")
        self.n_channels_unet = n_channels
        self.n_classes = n_classes; self.bilinear = bilinear
        self.activation_fn_template = copy.deepcopy(activation_fn)
        c1, c2, c3, c4 = 64, 128, 256, 512; c5 = 1024 # Bottleneck before halving for bilinear
        
        self.inc = DoubleConv(n_channels, c1, activation_fn=self.activation_fn_template)
        self.down1 = Down(c1, c2, activation_fn=self.activation_fn_template)
        self.down2 = Down(c2, c3, activation_fn=self.activation_fn_template)
        self.down3 = Down(c3, c4, activation_fn=self.activation_fn_template)
        factor = 2 if bilinear else 1
        self.down4 = Down(c4, c5 // factor, activation_fn=self.activation_fn_template) # Output is c5 // factor

        self.up1 = Up(c5 // factor, c4, c4 // factor, self.activation_fn_template, bilinear)
        self.up2 = Up(c4 // factor, c3, c3 // factor, self.activation_fn_template, bilinear)
        self.up3 = Up(c3 // factor, c2, c2 // factor, self.activation_fn_template, bilinear)
        self.up4 = Up(c2 // factor, c1, c1, self.activation_fn_template, bilinear) # Output c1
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)
        self.final_activation = nn.Tanh() # Ensure output is in [-1, 1] range before denormalization
    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.n_channels_unet:
            raise ValueError(f"UNet Input shape mismatch. Expected (N, {self.n_channels_unet}, H, W), got {x.shape}")
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits = self.outc(x); output = self.final_activation(logits)
        return output

# --- EDSR model for Deblurring ---
def make_edsr_activation_instance(act_template_fn, num_features):
    if isinstance(act_template_fn, FReLU): return FReLU(channels=num_features)
    elif isinstance(act_template_fn, SAG): return SAG(channels=num_features)
    elif isinstance(act_template_fn, CST_SAGA): return CST_SAGA(channels=num_features)
    elif isinstance(act_template_fn, nn.PReLU): return nn.PReLU(num_parameters=1) # Standard PReLU
    else: return copy.deepcopy(act_template_fn)

class ResidualBlock_EDSR(nn.Module):
    def __init__(self, n_feats, kernel_size, activation_fn_template, res_scale=0.1):
        super(ResidualBlock_EDSR, self).__init__()
        self.res_scale = res_scale
        current_activation = make_edsr_activation_instance(activation_fn_template, n_feats)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)),
            current_activation,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2))
        )
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR_Deblur(nn.Module):
    def __init__(self, n_channels_in=N_CHANNELS, n_channels_out=N_CHANNELS,
                 n_resblocks=EDSR_N_RESBLOCKS, n_feats=EDSR_N_FEATS,
                 activation_fn_template=nn.ReLU(), res_scale=EDSR_RES_SCALE):
        super(EDSR_Deblur, self).__init__()
        kernel_size = 3
        self.model_n_channels_in = n_channels_in
        self.head = nn.Conv2d(n_channels_in, n_feats, kernel_size, padding=(kernel_size//2))
        m_body = [
            ResidualBlock_EDSR(n_feats, kernel_size, activation_fn_template, res_scale)
            for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*m_body)
        self.body_skip_conv = nn.Conv2d(n_feats, n_feats, kernel_size=1) # Added for EDSR-style skip
        self.tail = nn.Conv2d(n_feats, n_channels_out, kernel_size, padding=(kernel_size//2))
        self.final_activation = nn.Tanh() # Ensure output is in [-1, 1] range before denormalization
    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.model_n_channels_in:
             raise ValueError(f"EDSR Input shape mismatch. Expected (N, {self.model_n_channels_in}, H, W), got {x.shape}")
        x_head = self.head(x)
        res_body = self.body(x_head)
        res_body_processed = self.body_skip_conv(res_body) # Process before adding skip
        res_body_processed += x_head # Skip connection from head to after body
        x_out = self.tail(res_body_processed)
        if hasattr(self, 'final_activation') and self.final_activation is not None:
            x_out = self.final_activation(x_out)
        return x_out

# --- Evaluation Metrics ---
def _convert_to_numpy(tensor: torch.Tensor, target_channels_for_metric: int) -> np.ndarray | None:
    if tensor is None or not isinstance(tensor, torch.Tensor): return None
    try:
        if tensor.dim() == 4: tensor = tensor.squeeze(0) # Remove batch if present
        tensor = tensor.detach().cpu().float() # Ensure on CPU, float type
        # Denormalize from [-1, 1] to [0, 1]
        img_np = tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1).astype(np.float32) # Clip to ensure valid range

        current_np_channels = img_np.shape[2]
        if current_np_channels == 1 and target_channels_for_metric == 3: # Convert Gray to RGB if needed
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif current_np_channels == 3 and target_channels_for_metric == 1: # Convert RGB to Gray if needed
            img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = np.expand_dims(img_np_gray, axis=2) # Keep 3D shape (H, W, 1)
        return img_np
    except Exception as e: print(f"Error (_convert_to_numpy): {e}"); return None

def calculate_epi(original_tensor: torch.Tensor, restored_tensor: torch.Tensor, window_size: int = 5) -> float:
    original_np_gray = _convert_to_numpy(original_tensor, target_channels_for_metric=1)
    restored_np_gray = _convert_to_numpy(restored_tensor, target_channels_for_metric=1)
    if original_np_gray is None or restored_np_gray is None or original_np_gray.shape[2]!=1 or restored_np_gray.shape[2]!=1: return float('nan')
    original_gray = original_np_gray.squeeze(axis=2); restored_gray = restored_np_gray.squeeze(axis=2)
    grad_x_orig = cv2.Sobel(original_gray, cv2.CV_32F, 1, 0, ksize=3); grad_y_orig = cv2.Sobel(original_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
    grad_x_res = cv2.Sobel(restored_gray, cv2.CV_32F, 1, 0, ksize=3); grad_y_res = cv2.Sobel(restored_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_res = np.sqrt(grad_x_res**2 + grad_y_res**2)
    pad = window_size // 2; grad_orig_pad = np.pad(grad_orig, pad, mode='reflect'); grad_res_pad = np.pad(grad_res, pad, mode='reflect')
    epi_values = []; rows, cols = grad_orig.shape
    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            window_orig = grad_orig_pad[i-pad:i+pad+1, j-pad:j+pad+1]; window_res = grad_res_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            mean_orig = np.mean(window_orig); mean_res = np.mean(window_res)
            cov = np.sum((window_orig - mean_orig) * (window_res - mean_res))
            var_orig_sq = np.sum((window_orig - mean_orig)**2); var_res_sq = np.sum((window_res - mean_res)**2)
            denominator = np.sqrt(var_orig_sq * var_res_sq)
            if denominator > 1e-9: epi = np.clip(cov / denominator, -1.0, 1.0); epi_values.append(epi)
    return np.mean(epi_values) if epi_values else 0.0

def calculate_hf_energy_ratio(original_tensor: torch.Tensor, restored_tensor: torch.Tensor) -> float:
    def get_hf_energy(img_tensor_for_hf):
        img_np_gray_for_hf = _convert_to_numpy(img_tensor_for_hf, target_channels_for_metric=1)
        if img_np_gray_for_hf is None or img_np_gray_for_hf.shape[2] != 1: return torch.tensor(0.0)
        img_gray_torch = torch.from_numpy(img_np_gray_for_hf.squeeze(axis=2)).cpu()
        fft = torch.fft.fft2(img_gray_torch.float()); fft_shift = torch.fft.fftshift(fft)
        h, w = img_gray_torch.shape; cy, cx = h // 2, w // 2; radius_ratio = 0.1
        radius = radius_ratio * min(cx, cy)
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=fft_shift.device)-cy, torch.arange(w, device=fft_shift.device)-cx, indexing='ij')
        mask = (x_coords**2+y_coords**2) > (radius**2)
        hf_energy = torch.sum(torch.abs(fft_shift) * mask)
        return hf_energy
    hf_original = get_hf_energy(original_tensor); hf_restored = get_hf_energy(restored_tensor)
    ratio = hf_restored / (hf_original + 1e-9)
    return ratio.item()

lpips_model_global = None; lpips_model_failed = False
def calculate_all_metrics(original_tensor: torch.Tensor, restored_tensor: torch.Tensor, device_metric: torch.device) -> dict:
    global lpips_model_global, lpips_model_failed
    original_np = _convert_to_numpy(original_tensor, target_channels_for_metric=N_CHANNELS) # Use global N_CHANNELS
    restored_np = _convert_to_numpy(restored_tensor, target_channels_for_metric=N_CHANNELS)
    if original_np is None or restored_np is None:
        return {"psnr": float('nan'), "ssim": float('nan'), "epi": float('nan'), "hf_recon": float('nan'), "lpips": float('nan')}
    psnr_val = float('nan'); ssim_val = float('nan')
    try: psnr_val = psnr(original_np, restored_np, data_range=1.0)
    except Exception: pass # Ignore errors like different shapes for robustness
    try:
        ssim_input_orig = original_np; ssim_input_rest = restored_np
        current_channel_axis_for_ssim = None
        if N_CHANNELS == 1: # If original was 1 channel, converted numpy is (H,W,1)
            if original_np.ndim == 3 and original_np.shape[2] == 1:
                ssim_input_orig = original_np.squeeze(axis=2) # SSIM expects (H,W) for grayscale
                ssim_input_rest = restored_np.squeeze(axis=2)
        elif N_CHANNELS > 1: current_channel_axis_for_ssim = 2 # For (H,W,C)

        if ssim_input_orig.shape != ssim_input_rest.shape: raise ValueError(f"Shape mismatch for SSIM: {ssim_input_orig.shape} vs {ssim_input_rest.shape}")

        min_spatial_dim = min(ssim_input_orig.shape[0], ssim_input_orig.shape[1])
        win_size = min(7, min_spatial_dim) # Default skimage win_size is 7
        if win_size < 3 : win_size = 3 if min_spatial_dim >=3 else min_spatial_dim # Ensure win_size is at least 3 or min_dim
        if win_size % 2 == 0: win_size -= 1 # Must be odd
        win_size = max(3, win_size) # Final check
        if win_size > min_spatial_dim : win_size = min_spatial_dim if min_spatial_dim % 2 == 1 else max(3, min_spatial_dim -1)
        if win_size <3 and min_spatial_dim >=3 : win_size =3 # last resort for small images

        ssim_val = ssim(ssim_input_orig, ssim_input_rest, data_range=1.0, win_size=win_size, channel_axis=current_channel_axis_for_ssim)
    except ValueError: pass # Likely shape mismatch or win_size issue
    except Exception: pass # Catch any other ssim error
    epi_val = calculate_epi(original_tensor, restored_tensor) # Uses original tensors
    hf_ratio_val = calculate_hf_energy_ratio(original_tensor, restored_tensor) # Uses original tensors
    lpips_val = float('nan')
    if LPIPS_AVAILABLE and not lpips_model_failed:
        if lpips_model_global is None:
            try: lpips_model_global = lpips.LPIPS(net='alex').to(device_metric); lpips_model_global.eval()
            except Exception as e: print(f"ERROR (LPIPS init): {e}"); lpips_model_failed = True
        if not lpips_model_failed:
            # LPIPS expects NCHW, [-1, 1] range. Our tensors are already NCHW and [-1,1] from model output / normalize
            img0_lpips = original_tensor.unsqueeze(0).to(device_metric) # Add batch dim
            img1_lpips = restored_tensor.unsqueeze(0).to(device_metric) # Add batch dim
            if N_CHANNELS == 1: # LPIPS AlexNet expects 3 channels
                img0_lpips = img0_lpips.repeat(1, 3, 1, 1)
                img1_lpips = img1_lpips.repeat(1, 3, 1, 1)
            with torch.no_grad():
                try: lpips_val = lpips_model_global(img0_lpips, img1_lpips).item()
                except Exception as e: print(f"LPIPS calc error: {e}") # Catch errors during calculation
    return {"psnr": psnr_val, "ssim": ssim_val, "epi": epi_val, "hf_recon": hf_ratio_val, "lpips": lpips_val}

# --- Activation analyzer class ---
class ActivationAnalyzer:
    def __init__(self, model):
        self.model = model; self.pre_act_maps = defaultdict(list); self.post_act_maps = defaultdict(list); self.hooks = []
    def _pre_act_hook(self, name, module, inp):
        if isinstance(inp, tuple): inp = inp[0]
        if inp is not None and isinstance(inp, torch.Tensor) and inp.nelement() > 0: self.pre_act_maps[name].append(inp[0].detach().cpu()) # Store first image in batch
    def _post_act_hook(self, name, module, inp, out):
        if isinstance(out, tuple): out = out[0]
        if out is not None and isinstance(out, torch.Tensor) and out.nelement() > 0: self.post_act_maps[name].append(out[0].detach().cpu()) # Store first image in batch
    def register_hooks(self):
        self.remove_hooks(); self.pre_act_maps.clear(); self.post_act_maps.clear();
        target_act_types = ( nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.SiLU, Mish, Swish, ESwish, Aria, GCU, Snake, FReLU, FAATanh, SAG, CST_SAGA )
        for name, module in self.model.named_modules():
            if isinstance(module, target_act_types):
                pre_hook = module.register_forward_pre_hook(partial(self._pre_act_hook, name)); post_hook = module.register_forward_hook(partial(self._post_act_hook, name))
                self.hooks.extend([pre_hook, post_hook])
    def remove_hooks(self): [hook.remove() for hook in self.hooks]; self.hooks = []
    def analyze_batch(self, input_batch): # Expects a single image tensor (CHW) or a batch (NCHW)
        if input_batch is None or input_batch.nelement() == 0: return
        self.model.eval();
        if input_batch.dim() == 3: input_batch = input_batch.unsqueeze(0) # Ensure batch dim for model
        with torch.no_grad():
            device_type = next(self.model.parameters()).device.type
            with autocast(enabled=(device_type == 'cuda')): _ = self.model(input_batch.to(next(self.model.parameters()).device))
    def get_activation_maps(self, layer_name): pre = self.pre_act_maps.get(layer_name, [None])[0]; post = self.post_act_maps.get(layer_name, [None])[0]; return pre, post # Return the first (and only) map

# --- Plotting functions ---
def save_feature_maps(pre_act, post_act, act_name, layer_name, run_idx, channel=0, result_dir=RESULT_DIR):
    if pre_act is None or post_act is None or pre_act.dim() < 2 or post_act.dim() < 2: return # Min 2D (HW)
    # Ensure CHW format if it's HW
    if pre_act.dim() == 2: pre_act = pre_act.unsqueeze(0) # Add channel dim
    if post_act.dim() == 2: post_act = post_act.unsqueeze(0) # Add channel dim

    # Now pre_act, post_act are CHW
    if pre_act.numel() == 0 or post_act.numel() == 0 or pre_act.shape[0] == 0 or post_act.shape[0] == 0: return
    num_channels_pre = pre_act.shape[0]; num_channels_post = post_act.shape[0]
    channel_to_plot = min(channel, num_channels_pre - 1, num_channels_post - 1) # Ensure valid channel
    if channel_to_plot < 0: return

    if pre_act.shape[-2:] != post_act.shape[-2:]: print(f"Warn: Mismatched spatial dimensions for layer {layer_name}. Skipping feature map."); return

    pre_map_np = pre_act[channel_to_plot].numpy(); post_map_np = post_act[channel_to_plot].numpy(); diff_map_np = post_map_np - pre_map_np
    plt.figure(figsize=(15, 5)); plt.clf()
    plt.subplot(131); im1 = plt.imshow(pre_map_np, cmap='viridis'); plt.title(f"Pre-Act ({layer_name}, C{channel_to_plot})"); plt.colorbar(im1); plt.axis('off')
    plt.subplot(132); im2 = plt.imshow(post_map_np, cmap='viridis'); plt.title(f"Post-Act ({act_name})"); plt.colorbar(im2); plt.axis('off')

    vmin_diff, vmax_diff = -1, 1 # Default range for diff
    if np.isfinite(diff_map_np).all() and diff_map_np.size > 1:
        try:
            vmin_calc, vmax_calc = np.percentile(diff_map_np, [1, 99])
            vmin_diff = np.min(diff_map_np) if vmin_calc >= vmax_calc else vmin_calc # Robust min
            vmax_diff = np.max(diff_map_np) if vmin_calc >= vmax_calc else vmax_calc # Robust max
        except Exception: vmin_diff, vmax_diff = np.min(diff_map_np), np.max(diff_map_np)
        if vmin_diff == vmax_diff: vmin_diff -= 0.5; vmax_diff += 0.5 # Avoid zero range for colorbar
    elif diff_map_np.size == 1: vmin_diff, vmax_diff = diff_map_np.item() - 0.5, diff_map_np.item() + 0.5

    plt.subplot(133); im3 = plt.imshow(diff_map_np, cmap='coolwarm', vmin=vmin_diff, vmax=vmax_diff); plt.title("Difference (Post - Pre)"); plt.colorbar(im3); plt.axis('off')
    save_dir = os.path.join(result_dir, "spatial_analysis", act_name); os.makedirs(save_dir, exist_ok=True)
    layer_name_safe = layer_name.replace('.','_').replace(':','_')
    save_path = os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_run{run_idx+1}_{act_name}_{layer_name_safe}_C{channel_to_plot}.png")
    try: plt.suptitle(f"Feature Map Analysis: Run {run_idx+1}", fontsize=10); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save feature map {save_path}: {e}")
    finally: plt.close()

def save_comparison_plot(blurred_tensor, restored_tensor, original_tensor, act_name, run_idx, metrics_summary=None, result_dir=RESULT_DIR):
    display_channels = 3 if N_CHANNELS == 1 else N_CHANNELS # Prefer 3 channels for display if input is grayscale
    blurred_np = _convert_to_numpy(blurred_tensor, target_channels_for_metric=display_channels)
    restored_np = _convert_to_numpy(restored_tensor, target_channels_for_metric=display_channels)
    original_np = _convert_to_numpy(original_tensor, target_channels_for_metric=display_channels)

    if blurred_np is None or restored_np is None or original_np is None: print("Warning: Could not prepare images for comparison plot."); return

    cmap_val = None
    if display_channels == 1 and blurred_np.ndim == 3 and blurred_np.shape[2] == 1: # (H,W,1)
        cmap_val = 'gray'
        blurred_np = blurred_np.squeeze(axis=2)
        restored_np = restored_np.squeeze(axis=2)
        original_np = original_np.squeeze(axis=2)
    elif display_channels == 3 and blurred_np.ndim == 2: # (H,W) but should be (H,W,3)
         print("Warning: Expected 3 channels for display, got 2D array. Plotting as gray.")
         cmap_val = 'gray'

    metric_str = "";
    if metrics_summary:
        items = []
        key_precision_map = {"psnr_mean": ".2f", "ssim_mean": ".4f", "epi_mean": ".4f", "hf_recon_mean": ".4f", "lpips_mean": ".4f"}
        for k, v_metric in metrics_summary.items():
            if k in key_precision_map and not np.isnan(v_metric):
                fmt_str = key_precision_map[k]; metric_name_display = k.replace('_mean','').upper()
                if metric_name_display == "HF_RECON": metric_name_display = "HF-R"
                items.append(f"{metric_name_display}: {v_metric:{fmt_str}}")
        if items: metric_str = "\n(Avg Val Metrics) " + " | ".join(items)

    plt.figure(figsize=(18, 6)); plt.clf(); plt.suptitle(f"{MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} - Run {run_idx+1}", fontsize=14)
    plt.subplot(131); plt.imshow(blurred_np, cmap=cmap_val); plt.title("Degraded Input"); plt.axis('off')
    plt.subplot(132); plt.imshow(restored_np, cmap=cmap_val); plt.title(f"{act_name} Restored {metric_str}"); plt.axis('off')
    plt.subplot(133); plt.imshow(original_np, cmap=cmap_val); plt.title("Ground Truth"); plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_dir = os.path.join(result_dir, "comparison_plots"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_comparison.png")
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save comparison plot {save_path}: {e}")
    finally: plt.close()

def plot_loss_curves(train_losses, val_losses, act_name, run_idx, result_dir=RESULT_DIR):
    epochs_range = range(1, len(train_losses) + 1); plt.figure(figsize=(10, 6)); plt.clf()
    plt.plot(epochs_range, train_losses, label='Training Loss', linewidth=2); plt.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'{MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} Loss ({act_name} Run {run_idx+1})', fontsize=14); plt.xlabel('Epoch', fontsize=12); plt.ylabel('Loss (L1)', fontsize=12)
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3)
    valid_losses = [l for l in train_losses + val_losses if l is not None and np.isfinite(l)]
    if valid_losses: min_loss = min(valid_losses); max_loss = max(valid_losses); plt.ylim(bottom=max(0, min_loss-0.1*(max_loss-min_loss)), top=max_loss+0.1*(max_loss-min_loss))
    else: plt.ylim(bottom=0)
    save_dir = os.path.join(result_dir, "loss_curves"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_loss_curves.png")
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save loss curve {save_path}: {e}")
    finally: plt.close()

def analyze_frequency_and_spatial(model, analyzer, gt_img_tensor, pred_img_tensor, blurred_img_tensor, act_name, run_idx, result_dir=RESULT_DIR):
    device_freq = next(model.parameters()).device
    def prepare_spectrum_from_tensor(img_tensor_in): # Expects CHW tensor
        img_np_gray = _convert_to_numpy(img_tensor_in.cpu(), target_channels_for_metric=1) # Convert to (H,W,1) numpy
        return torch.from_numpy(img_np_gray.squeeze(axis=2)) if img_np_gray is not None and img_np_gray.shape[2]==1 else None
    def compute_spectrum_log(img_gray_torch_tensor): # Expects HW tensor
        return torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(img_gray_torch_tensor.float()))) + 1e-9) if img_gray_torch_tensor is not None else None
    print(f"Analyzing frequency domain for {act_name} Run {run_idx+1}...")
    gt_spectrum = compute_spectrum_log(prepare_spectrum_from_tensor(gt_img_tensor))
    blurred_spectrum = compute_spectrum_log(prepare_spectrum_from_tensor(blurred_img_tensor))
    pred_spectrum = compute_spectrum_log(prepare_spectrum_from_tensor(pred_img_tensor))
    if gt_spectrum is not None and blurred_spectrum is not None and pred_spectrum is not None:
        plt.figure(figsize=(18, 6)); plt.clf(); plt.suptitle(f"{MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} Frequency Spectra - Run {run_idx+1}", fontsize=14)
        plt.subplot(131); plt.imshow(gt_spectrum.numpy(), cmap='viridis'); plt.title("Ground Truth Spectrum"); plt.axis('off')
        plt.subplot(132); plt.imshow(blurred_spectrum.numpy(), cmap='viridis'); plt.title("Degraded Spectrum"); plt.axis('off')
        plt.subplot(133); plt.imshow(pred_spectrum.numpy(), cmap='viridis'); plt.title(f"{act_name} Restored Spectrum"); plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_dir = os.path.join(result_dir, "fft_analysis"); os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name}_run{run_idx+1}_main_spectra.png")
        try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
        except Exception as e: print(f" Failed to save spectra plot {save_path}: {e}")
        finally: plt.close()
    else: print("Skipping frequency plot due to issue preparing spectra.")

    # Spatial analysis is intensive and model-specific (UNet layers)
    if MODEL_CHOICE == "UNET": # Or specific layers for EDSR if desired
        print(f"Analyzing spatial domain (feature maps) for {act_name} Run {run_idx+1}...")
        analyzer.register_hooks()
        analyzer.analyze_batch(blurred_img_tensor.to(device_freq)) # Pass single CHW tensor
        analyzer.remove_hooks()
        layers_to_visualize = list(analyzer.pre_act_maps.keys())
        if not layers_to_visualize: print("No activation layers hooked for spatial analysis."); return
        if len(layers_to_visualize) > 6: layers_to_visualize = random.sample(layers_to_visualize, 6)
        print(f"Visualizing feature maps for {len(layers_to_visualize)} layers: {layers_to_visualize}")
        for layer_name in layers_to_visualize:
            pre_map, post_map = analyzer.get_activation_maps(layer_name) # These are CHW
            if pre_map is not None and post_map is not None and pre_map.ndim ==3 and pre_map.shape[-1] > 1 and pre_map.shape[-2] > 1: # Check if CHW and spatial dims > 1
                 save_feature_maps(pre_map, post_map, act_name, layer_name, run_idx, channel=0, result_dir=result_dir)
    else:
        print(f"Skipping detailed spatial feature map analysis for {MODEL_CHOICE} (analyzer hooks may need adaptation or is not UNet).")

# --- HPO objective function ---
def objective(trial, act_name_hpo, act_config_hpo, hpo_train_loader, hpo_val_loader, device_hpo):
    lr_hpo = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    weight_decay_hpo = GLOBAL_WEIGHT_DECAY # Could also be optimized
    activation_fn_hpo_template = copy.deepcopy(act_config_hpo["fn"])
    if MODEL_CHOICE == "EDSR":
        model_hpo = EDSR_Deblur(activation_fn_template=activation_fn_hpo_template).to(device_hpo)
    else: # UNet
        model_hpo = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=activation_fn_hpo_template).to(device_hpo)
    optimizer_hpo = optim.Adam(model_hpo.parameters(), lr=lr_hpo, weight_decay=weight_decay_hpo)
    criterion_hpo = nn.L1Loss(); scaler_hpo = GradScaler(enabled=(device_hpo.type == 'cuda'))
    model_hpo.train()
    for epoch_hpo in range(HPO_N_EPOCHS):
        num_batch_train_hpo = 0
        for batch_data_hpo in hpo_train_loader:
            degraded_hpo, sharp_hpo, _ = batch_data_hpo # Unpack image_id
            degraded_hpo, sharp_hpo = degraded_hpo.to(device_hpo), sharp_hpo.to(device_hpo)
            optimizer_hpo.zero_grad(set_to_none=True)
            with autocast(enabled=(device_hpo.type == 'cuda')): output_hpo = model_hpo(degraded_hpo); loss_hpo = criterion_hpo(output_hpo, sharp_hpo)
            if torch.isnan(loss_hpo) or torch.isinf(loss_hpo): print(f"HPO Trial {trial.number} pruned: NaN/Inf train loss."); raise optuna.TrialPruned()
            scaler_hpo.scale(loss_hpo).backward(); scaler_hpo.step(optimizer_hpo); scaler_hpo.update()
            num_batch_train_hpo += 1
            if num_batch_train_hpo >= 10: break # Limit batches for HPO speed
    model_hpo.eval(); val_loss_hpo_accum = 0; num_batches_hpo_val = 0
    with torch.no_grad():
        for batch_data_val_hpo in hpo_val_loader:
            degraded_val_hpo, sharp_val_hpo, _ = batch_data_val_hpo # Unpack image_id
            degraded_val_hpo, sharp_val_hpo = degraded_val_hpo.to(device_hpo), sharp_val_hpo.to(device_hpo)
            with autocast(enabled=(device_hpo.type == 'cuda')): output_val_hpo = model_hpo(degraded_val_hpo); v_loss_hpo = criterion_hpo(output_val_hpo, sharp_val_hpo).item()
            if not np.isnan(v_loss_hpo) and not np.isinf(v_loss_hpo): val_loss_hpo_accum += v_loss_hpo; num_batches_hpo_val += 1
            if num_batches_hpo_val >= 5: break # Limit batches for HPO speed
    avg_val_loss_hpo = val_loss_hpo_accum / num_batches_hpo_val if num_batches_hpo_val > 0 else float('inf')
    if np.isnan(avg_val_loss_hpo) or avg_val_loss_hpo == float('inf'): print(f"HPO Trial {trial.number} pruned: NaN/Inf val loss."); raise optuna.TrialPruned()
    trial.report(avg_val_loss_hpo, HPO_N_EPOCHS - 1) # Report at the end of HPO epochs
    if trial.should_prune(): print(f"HPO Trial {trial.number} pruned by Optuna."); raise optuna.TrialPruned()
    return avg_val_loss_hpo

# --- Training & validation function ---
def train_and_evaluate(act_name_train, act_config_train, learning_rate_train, weight_decay_train,
                       train_loader_main, val_loader_main, run_idx_train, device_train, result_dir_train=RESULT_DIR, model_dir_train=MODEL_DIR):
    print(f"\n=== Training {MODEL_CHOICE} Run {run_idx_train+1} with {act_name_train} (LR={learning_rate_train:.2e}) ===")
    current_activation_fn_template = copy.deepcopy(act_config_train["fn"])
    if MODEL_CHOICE == "EDSR":
        model_train = EDSR_Deblur(activation_fn_template=current_activation_fn_template).to(device_train)
        print(f"Instantiated EDSR_Deblur with n_resblocks={EDSR_N_RESBLOCKS}, n_feats={EDSR_N_FEATS}")
    else: # UNet
        model_train = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=current_activation_fn_template).to(device_train)
        print("Instantiated UNet")
    analyzer_train = ActivationAnalyzer(model_train)
    optimizer_train = optim.Adam(model_train.parameters(), lr=learning_rate_train, weight_decay=weight_decay_train)
    scheduler_train = ReduceLROnPlateau(optimizer_train, 'min', patience=7, factor=0.5, verbose=False) # verbose=True can be useful
    criterion_train = nn.L1Loss(); use_amp_train = (device_train.type == 'cuda'); scaler_train = GradScaler(enabled=use_amp_train)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp_train}")
    train_loss_history_ep = []; val_loss_history_ep = []
    best_val_loss_ep = float('inf'); epochs_no_improve_ep = 0; best_model_state_ep = None
    vis_blurred_val_tensor, vis_sharp_val_tensor = None, None # Tensors for visualization
    try:
        # Get a fixed batch for visualization from the validation set (not shuffled)
        vis_data_loader = DataLoader(val_loader_main.dataset, batch_size=1, shuffle=False) # Use a new loader to not interfere
        vis_batch_content = next(iter(vis_data_loader))
        vis_blurred_val_tensor, vis_sharp_val_tensor = vis_batch_content[0][0], vis_batch_content[1][0] # Get first image (CHW)
    except StopIteration: print("Warning: Could not get visualization batch from validation loader.")
    except Exception as e_vis: print(f"Warning: Error getting visualization batch: {e_vis}")

    for epoch_train in range(EPOCHS):
        model_train.train(); epoch_loss_accum = 0; num_train_batches_ep = 0
        pbar_train_ep = tqdm(train_loader_main, desc=f"Epoch {epoch_train+1}/{EPOCHS} [Train]", leave=False)
        for batch_data_train_ep in pbar_train_ep:
            blurred_ep, sharp_ep, _ = batch_data_train_ep # Unpack image_id
            blurred_ep, sharp_ep = blurred_ep.to(device_train), sharp_ep.to(device_train)
            optimizer_train.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp_train): output_ep = model_train(blurred_ep); loss_ep = criterion_train(output_ep, sharp_ep)
            if torch.isnan(loss_ep) or torch.isinf(loss_ep): print(f"WARNING: NaN/Inf train loss epoch {epoch_train+1}. Skipping batch."); continue
            scaler_train.scale(loss_ep).backward(); scaler_train.unscale_(optimizer_train); torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0); scaler_train.step(optimizer_train); scaler_train.update()
            epoch_loss_accum += loss_ep.item(); num_train_batches_ep += 1; pbar_train_ep.set_postfix(loss=f"{loss_ep.item():.4f}")
        avg_train_loss_ep = epoch_loss_accum / num_train_batches_ep if num_train_batches_ep > 0 else float('nan'); train_loss_history_ep.append(avg_train_loss_ep)
        model_train.eval(); val_loss_accum_ep = 0; num_val_batches_ep = 0; epoch_val_metrics_lists = defaultdict(list)
        pbar_val_ep = tqdm(val_loader_main, desc=f"Epoch {epoch_train+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch_data_val_ep in pbar_val_ep:
                blurred_val_ep, sharp_val_ep, _ = batch_data_val_ep # Unpack image_id
                blurred_val_ep, sharp_val_ep = blurred_val_ep.to(device_train), sharp_val_ep.to(device_train)
                with autocast(enabled=use_amp_train): output_val_ep = model_train(blurred_val_ep); v_loss_ep = criterion_train(output_val_ep, sharp_val_ep)
                if not torch.isnan(v_loss_ep) and not torch.isinf(v_loss_ep):
                    val_loss_accum_ep += v_loss_ep.item(); num_val_batches_ep += 1
                    # Calculate metrics on the first image of the validation batch for quick check
                    if blurred_val_ep.size(0) > 0:
                         metrics_batch_first_img = calculate_all_metrics(sharp_val_ep[0].float(), output_val_ep[0].float(), device_train)
                         for key_metric, val_metric in metrics_batch_first_img.items(): epoch_val_metrics_lists[key_metric].append(val_metric)
                else: print(f"WARNING: NaN/Inf val loss epoch {epoch_train+1}. Skipping batch.")
                pbar_val_ep.set_postfix(loss=f"{v_loss_ep.item() if not (torch.isnan(v_loss_ep) or torch.isinf(v_loss_ep)) else float('nan'):.4f}")
        avg_val_loss_ep = val_loss_accum_ep / num_val_batches_ep if num_val_batches_ep > 0 else float('inf'); val_loss_history_ep.append(avg_val_loss_ep)
        avg_epoch_val_metrics_print = {}
        for key_m, list_m_vals in epoch_val_metrics_lists.items():
            valid_m_vals = [m for m in list_m_vals if m is not None and np.isfinite(m)]
            avg_epoch_val_metrics_print[f"{key_m}_mean"] = np.mean(valid_m_vals) if valid_m_vals else float('nan')
        print(f"Epoch {epoch_train+1}/{EPOCHS} - Train Loss: {avg_train_loss_ep:.5f}, Val Loss: {avg_val_loss_ep:.5f} "
              f"(Val PSNR: {avg_epoch_val_metrics_print.get('psnr_mean', 0.0):.2f}, "
              f"Val SSIM: {avg_epoch_val_metrics_print.get('ssim_mean', 0.0):.4f}) "
              f"(LR: {optimizer_train.param_groups[0]['lr']:.2e})")
        scheduler_train.step(avg_val_loss_ep)
        if avg_val_loss_ep < best_val_loss_ep:
            print(f"  Val loss improved ({best_val_loss_ep:.5f} -> {avg_val_loss_ep:.5f}). Saving model...");
            best_val_loss_ep = avg_val_loss_ep; best_model_state_ep = copy.deepcopy(model_train.state_dict()); epochs_no_improve_ep = 0
            save_path_model = os.path.join(model_dir_train, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name_train}_run{run_idx_train+1}_best.pth")
            torch.save(best_model_state_ep, save_path_model)
        else: epochs_no_improve_ep += 1
        if epochs_no_improve_ep >= EARLY_STOPPING_PATIENCE: print(f"Early stopping @ epoch {epoch_train + 1}."); break
    print("Training finished."); plot_loss_curves(train_loss_history_ep, val_loss_history_ep, act_name_train, run_idx_train, result_dir=result_dir_train)
    if best_model_state_ep is None: print("Warning: No best model state saved. Using last model state."); best_model_state_ep = model_train.state_dict()
    model_train.load_state_dict(best_model_state_ep); model_train.eval()
    print("Calculating final metrics on full validation set using best model...")
    final_val_metrics_all_imgs = defaultdict(list); final_val_loss_accum = 0; final_val_batches_count = 0
    with torch.no_grad():
        for batch_data_final_val in val_loader_main:
            blurred_final, sharp_final, _ = batch_data_final_val # Unpack image_id
            blurred_final, sharp_final = blurred_final.to(device_train), sharp_final.to(device_train)
            with autocast(enabled=use_amp_train): output_final = model_train(blurred_final); v_loss_final = criterion_train(output_final, sharp_final)
            if not torch.isnan(v_loss_final) and not torch.isinf(v_loss_final):
                final_val_loss_accum += v_loss_final.item(); final_val_batches_count += 1
                for j_img in range(blurred_final.size(0)):
                     metrics_per_img = calculate_all_metrics(sharp_final[j_img].float(), output_final[j_img].float(), device_train)
                     for key_metric_final, val_metric_final in metrics_per_img.items():
                         final_val_metrics_all_imgs[key_metric_final].append(val_metric_final)
    avg_final_val_metrics_summary = {}
    for key_sum, values_sum in final_val_metrics_all_imgs.items():
         valid_sum = [v for v in values_sum if v is not None and np.isfinite(v)]
         avg_final_val_metrics_summary[f"{key_sum}_mean"] = np.mean(valid_sum) if valid_sum else float('nan')
         avg_final_val_metrics_summary[f"{key_sum}_std"] = np.std(valid_sum) if len(valid_sum) > 1 else 0.0
    avg_final_val_loss_summary = final_val_loss_accum / final_val_batches_count if final_val_batches_count > 0 else float('inf')
    summary_results_dict = {
        "psnr_mean": avg_final_val_metrics_summary.get("psnr_mean", float('nan')), "psnr_std": avg_final_val_metrics_summary.get("psnr_std", float('nan')),
        "ssim_mean": avg_final_val_metrics_summary.get("ssim_mean", float('nan')), "ssim_std": avg_final_val_metrics_summary.get("ssim_std", float('nan')),
        "epi_mean": avg_final_val_metrics_summary.get("epi_mean", float('nan')), "epi_std": avg_final_val_metrics_summary.get("epi_std", float('nan')),
        "hf_recon_mean": avg_final_val_metrics_summary.get("hf_recon_mean", float('nan')), "hf_recon_std": avg_final_val_metrics_summary.get("hf_recon_std", float('nan')),
        "lpips_mean": avg_final_val_metrics_summary.get("lpips_mean", float('nan')), "lpips_std": avg_final_val_metrics_summary.get("lpips_std", float('nan')),
        "final_train_loss": train_loss_history_ep[-1] if train_loss_history_ep and np.isfinite(train_loss_history_ep[-1]) else float('nan'),
        "final_val_loss": avg_final_val_loss_summary if np.isfinite(avg_final_val_loss_summary) else float('nan'),
        "inf_time_ms": float('nan') # Placeholder for inference time if measured
    }
    if vis_blurred_val_tensor is not None and vis_sharp_val_tensor is not None:
        print("Generating visualization plot using best model...")
        with torch.no_grad():
            vis_input_tensor = vis_blurred_val_tensor.unsqueeze(0).to(device_train) # Add batch dim
            with autocast(enabled=use_amp_train): vis_output_val_tensor_dev = model_train(vis_input_tensor)
        # Pass CHW tensors (after removing batch dim for output, inputs are already CHW)
        save_comparison_plot(vis_blurred_val_tensor.cpu(), vis_output_val_tensor_dev[0].cpu(), vis_sharp_val_tensor.cpu(), act_name_train, run_idx_train, summary_results_dict, result_dir=result_dir_train)
        analyze_frequency_and_spatial(model_train, analyzer_train, vis_sharp_val_tensor.cpu(), vis_output_val_tensor_dev[0].cpu(), vis_blurred_val_tensor.cpu(), act_name_train, run_idx_train, result_dir=result_dir_train)
    print(f"--- {MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} {act_name_train} Run {run_idx_train+1} Final Validation Summary ---")
    keys_to_print_summary = [k for k in summary_results_dict.keys() if '_std' not in k]
    for key_p in keys_to_print_summary:
        value_p = summary_results_dict.get(key_p, float('nan')); metric_name_upper_p = key_p.replace('_mean','').replace('final_','').upper()
        if np.isfinite(value_p):
            if "_loss" not in key_p and "inf_time" not in key_p:
                 std_key_p = key_p.replace("_mean", "") + "_std"; std_val_p = summary_results_dict.get(std_key_p, float('nan'))
                 if np.isfinite(std_val_p): print(f"  {metric_name_upper_p}: {value_p:.4f} +/- {std_val_p:.4f}")
                 else: print(f"  {metric_name_upper_p}: {value_p:.4f}")
            else: print(f"  {metric_name_upper_p}: {value_p:.4f}")
        else: print(f"  {metric_name_upper_p}: nan")
    print("-------------------------------------")
    return summary_results_dict

# --- Evaluate on test set ---
def evaluate_model_on_test_set(model_path_test, activation_fn_instance_test, test_loader_main, device_test,
                               act_name_test, run_idx_test, model_architecture_name=MODEL_CHOICE,
                               dataset_name_test=DATASET_NAME, task_name_test=TASK_NAME):
    print(f"\n--- Evaluating {model_architecture_name} ({act_name_test}, Run {run_idx_test+1}) on **Test Set** ---")
    if not os.path.exists(model_path_test): print(f"ERROR: Model file not found: {model_path_test}. Skipping test eval."); return []
    try:
        if model_architecture_name == "EDSR":
            model_test = EDSR_Deblur(activation_fn_template=activation_fn_instance_test).to(device_test)
        else: # UNet
            model_test = UNet(n_channels=N_CHANNELS, n_classes=N_CHANNELS, activation_fn=activation_fn_instance_test).to(device_test)
        model_test.load_state_dict(torch.load(model_path_test, map_location=device_test)); model_test.eval()
        print(f"Successfully loaded best {model_architecture_name} model from: {model_path_test} for test set evaluation.")
    except Exception as e_load_test: print(f"ERROR: Failed load/instantiate {model_architecture_name} model from {model_path_test}: {e_load_test}"); traceback.print_exc(); return []
    use_amp_test = (device_test.type == 'cuda'); all_image_metrics_test = []
    with torch.no_grad():
        pbar_test_eval = tqdm(test_loader_main, desc=f"Testing {model_architecture_name} ({act_name_test} R{run_idx_test+1})", leave=False)
        for batch_data_test in pbar_test_eval:
            # Expecting (degraded_tensor, sharp_tensor, image_id) from dataset
            if len(batch_data_test) == 3: blurred_test, sharp_test, image_ids_test = batch_data_test
            else: # Fallback if image_ids are missing (should not happen with current dataset)
                blurred_test, sharp_test = batch_data_test
                image_ids_test = [f"placeholder_{i}" for i in range(blurred_test.size(0))]

            blurred_dev_test, sharp_dev_test = blurred_test.to(device_test), sharp_test.to(device_test)
            with autocast(enabled=use_amp_test): output_dev_test = model_test(blurred_dev_test)
            output_cpu_test = output_dev_test.cpu().float(); sharp_cpu_test = sharp_dev_test.cpu().float() # Move to CPU for metrics

            for i_test_img in range(blurred_test.size(0)):
                img_id_test = image_ids_test[i_test_img]
                target_img_tensor_test = sharp_cpu_test[i_test_img] # CHW tensor
                pred_img_tensor_test = output_cpu_test[i_test_img]   # CHW tensor
                metrics_test_img = calculate_all_metrics(target_img_tensor_test, pred_img_tensor_test, device_test) # Pass tensors
                image_result_test = {
                    "ImageID": img_id_test, "Architecture": model_architecture_name, "Dataset": dataset_name_test,
                    "Task": task_name_test, "ActivationFunction": act_name_test, "RunIndex": run_idx_test + 1,
                    "PSNR": metrics_test_img.get("psnr", float('nan')), "SSIM": metrics_test_img.get("ssim", float('nan')),
                    "EPI": metrics_test_img.get("epi", float('nan')), "HF_Score": metrics_test_img.get("hf_recon", float('nan')),
                    "LPIPS": metrics_test_img.get("lpips", float('nan'))
                }
                all_image_metrics_test.append(image_result_test)
    print(f"--- Test Set Evaluation Complete for {model_architecture_name} {act_name_test} Run {run_idx_test+1} ({len(all_image_metrics_test)} images processed) ---")
    return all_image_metrics_test
#----- Statistical Analysis------
# --- ANOVA analysis function ---
def run_anova_analysis(df_anova: pd.DataFrame, metrics_anova: list, result_dir_anova: str):
    activation_col_anova = "ActivationFunction"; print("\n" + "="*30 + " ANOVA Analysis on Test Set Results " + "="*30)
    if activation_col_anova not in df_anova.columns: print(f"ERROR: Column '{activation_col_anova}' not found in DataFrame."); return
    for metric_a in metrics_anova:
        print(f"\n--- ANOVA Analysis for Metric: {metric_a.upper()} ---")
        if metric_a not in df_anova.columns: print(f"  Metric column '{metric_a}' not found."); continue
        df_metric_anova = df_anova[[activation_col_anova, metric_a]].dropna()
        if df_metric_anova.empty: print(f"  No valid data for {metric_a} after dropping NaNs."); continue
        groups_anova = df_metric_anova[activation_col_anova].unique()
        if len(groups_anova) < 2: print(f"  Need at least 2 activation groups for ANOVA, found {len(groups_anova)} for {metric_a}. Skipping."); continue
        group_data_anova = [df_metric_anova[metric_a][df_metric_anova[activation_col_anova] == group].values for group in groups_anova]

        # Check if any group has less than 3 samples or zero variance (can cause issues with tests)
        if any(len(g) < 3 for g in group_data_anova): print(f"  Warning: Skipping ANOVA for {metric_a}: At least one group has fewer than 3 samples."); continue
        if any(np.var(g) < 1e-10 for g in group_data_anova if len(g) > 0): print(f"  Warning: Skipping ANOVA for {metric_a}: At least one group has zero variance."); continue

        print(f"  Checking Assumptions for {metric_a}:"); assumptions_met_anova = True
        try: # Normality of Residuals (using OLS model residuals)
            model_ols = ols(f'`{metric_a}` ~ C(`{activation_col_anova}`)', data=df_metric_anova).fit() # Backticks for special chars in metric names
            if len(model_ols.resid) >= 3: # Shapiro-Wilk needs at least 3 samples
                shapiro_test_stat, shapiro_p_val = stats.shapiro(model_ols.resid)
                print(f"    Normality of Residuals (Shapiro-Wilk): Statistic={shapiro_test_stat:.4f}, p-value={shapiro_p_val:.4f}")
                if shapiro_p_val < 0.05: print("    *Normality of residuals possibly violated (p < 0.05).*")
            else: print("    Skipping normality test on residuals (fewer than 3 residuals).")
        except Exception as e_norm: print(f"    Normality test (residuals) failed for {metric_a}: {e_norm}")

        try: # Homogeneity of Variances (Levene's Test)
             valid_group_data_for_levene = [g for g in group_data_anova if len(g) > 0 and np.var(g) > 1e-10] # Filter out empty or zero-variance groups for Levene
             if len(valid_group_data_for_levene) >= 2: # Levene's test needs at least 2 groups
                 stat_levene, p_levene = stats.levene(*valid_group_data_for_levene)
                 print(f"    Homogeneity of Variances (Levene's Test): Statistic={stat_levene:.4f}, p-value={p_levene:.4f}")
                 if p_levene < 0.05: print("    *Equal variances possibly violated (p < 0.05).*"); assumptions_met_anova = False # Mark assumption as not met
             else: print(f"    Skipping Levene's test (not enough valid groups for comparison).")
        except Exception as e_levene: print(f"    Levene's test failed for {metric_a}: {e_levene}")

        print(f"  Performing ANOVA for {metric_a}:")
        try:
            anova_p_value_report = float('nan')
            # Use stats.f_oneway for standard ANOVA
            f_stat_anova, p_anova_val = stats.f_oneway(*group_data_anova)
            print(f"    Standard One-Way ANOVA: F-statistic={f_stat_anova:.4f}, p-value={p_anova_val:.4f}")
            anova_p_value_report = p_anova_val

            anova_significant_report = anova_p_value_report < 0.05
            if anova_significant_report:
                print(f"  ANOVA is significant for {metric_a} (p < 0.05). Performing Tukey's HSD post-hoc test...")
                try:
                    tukey_result_obj = pairwise_tukeyhsd(endog=df_metric_anova[metric_a], groups=df_metric_anova[activation_col_anova], alpha=0.05)
                    print(tukey_result_obj)
                    # Save Tukey's HSD results to a file
                    tukey_summary_file_path = os.path.join(result_dir_anova, "metrics", f"{EXPERIMENT_FOLDER_NAME}_tukey_HSD_{metric_a}.txt")
                    with open(tukey_summary_file_path, "w") as f_tukey: f_tukey.write(str(tukey_result_obj))
                    print(f"    Tukey's HSD results saved to: {tukey_summary_file_path}")
                except Exception as e_tukey: print(f"    ERROR: Tukey's HSD failed for {metric_a}: {e_tukey}"); traceback.print_exc()
            else: print(f"  ANOVA is not significant for {metric_a} (p >= 0.05). No pairwise post-hoc tests needed.")
        except ValueError as ve_anova: print(f"  Skipping ANOVA for {metric_a} due to ValueError: {ve_anova}") # e.g. "Must be at least two groups"
        except Exception as e_anova: print(f"  Unexpected ANOVA error for {metric_a}: {e_anova}"); traceback.print_exc()
    print("\n" + "="*28 + " End ANOVA Analysis " + "="*28 + "\n")

# --- Main script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation Function Analysis for Medical Images")
    parser.add_argument("--job_part", type=int, default=None,
                        help="Specify which part of the activation functions to run (e.g., 1, 2). Runs all if not specified.")
    parser.add_argument("--total_jobs", type=int, default=2,
                        help="Total number of jobs the workload is split into.")
    parser.add_argument("--combine_results", action="store_true",
                        help="If set, skips training and combines results from previous partial runs.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", ".*torch.load with weights_only=False.*")
    warnings.filterwarnings("ignore", "The given NumPy array is not writable.*")
    warnings.filterwarnings("ignore", "torch.meshgrid.*")
    warnings.filterwarnings("ignore", ".*torch.amp.autocast.* is deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, module='skimage')
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
    if hasattr(torch.serialization, 'add_safe_globals'): # For older PyTorch versions
        torch.serialization.add_safe_globals([torch])


    # --- Determine activations to run for this Job Part ( to make the job run in GPU with less resourses) ---
    activations_to_run_this_job = {}
    if not args.combine_results:
        all_act_keys = list(ALL_ACTIVATIONS_CONFIG.keys())
        num_total_activations = len(all_act_keys)

        if args.job_part is not None:
            if not (1 <= args.job_part <= args.total_jobs):
                raise ValueError(f"job_part must be between 1 and total_jobs ({args.total_jobs}), got {args.job_part}")

            chunk_size = math.ceil(num_total_activations / args.total_jobs)
            start_idx = (args.job_part - 1) * chunk_size
            end_idx = min(start_idx + chunk_size, num_total_activations)
            selected_act_names_this_job = all_act_keys[start_idx:end_idx]

            for name in selected_act_names_this_job:
                activations_to_run_this_job[name] = ALL_ACTIVATIONS_CONFIG[name]
            print(f"INFO: Running Job Part {args.job_part}/{args.total_jobs} with {len(activations_to_run_this_job)} activations: {list(activations_to_run_this_job.keys())}")
        else:
            activations_to_run_this_job = ALL_ACTIVATIONS_CONFIG
            print(f"INFO: Running all {len(activations_to_run_this_job)} activations (no job_part specified).")
    else:
        print("INFO: Combine results mode. Skipping training and evaluation.")


    # --- Combine results ---
    if args.combine_results:
        print("\n" + "="*20 + " Combining Results and Running Final Analysis " + "="*20)
        # 1. Combine Per-Image Test Metrics for ANOVA
        all_combined_test_dfs = []
        for i in range(1, args.total_jobs + 1):
            part_suffix = f"_part{i}"
            partial_csv_path = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_per_image_test_metrics{part_suffix}.csv")
            if os.path.exists(partial_csv_path):
                print(f"Loading partial test metrics from: {partial_csv_path}")
                try:
                    df_part = pd.read_csv(partial_csv_path)
                    all_combined_test_dfs.append(df_part)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Partial test metrics file {partial_csv_path} is empty. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not load {partial_csv_path}: {e}")
            else:
                print(f"Warning: Partial test metrics file not found: {partial_csv_path}")

        final_combined_test_df = None
        if all_combined_test_dfs:
            final_combined_test_df = pd.concat(all_combined_test_dfs, ignore_index=True)
            combined_csv_output_path = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_COMBINED_per_image_test_metrics_FOR_ANOVA.csv")
            try:
                final_combined_test_df.to_csv(combined_csv_output_path, index=False, float_format='%.6f')
                print(f"Combined per-image test metrics (for ANOVA) saved to: {combined_csv_output_path}")

                print("\n--- Performing ANOVA Analysis on Combined Test Set Metrics ---")
                metrics_for_anova_final = ["PSNR", "SSIM", "EPI", "HF_Score", "LPIPS"] # Ensure these match CSV headers
                metrics_present_for_anova = [m for m in metrics_for_anova_final if m in final_combined_test_df.columns]
                missing_metrics_anova = set(metrics_for_anova_final) - set(metrics_present_for_anova)
                if missing_metrics_anova: print(f"Warning: Metrics not found in combined results for ANOVA: {missing_metrics_anova}")

                if metrics_present_for_anova:
                    run_anova_analysis(final_combined_test_df, metrics_present_for_anova, RESULT_DIR)
                else:
                    print("No valid metrics found in the combined DataFrame for ANOVA analysis.")
            except IOError as e_csv_io: print(f"\nError saving combined per-image results CSV: {e_csv_io}")
            except Exception as e_final_anova: print(f"\nError during final ANOVA analysis: {e_final_anova}"); traceback.print_exc()
        else:
            print("No partial test result files found to combine for ANOVA.")

        # 2. Combine aggregated validation summary
        print("\n" + "="*20 + " Combining Aggregated Validation Summaries " + "="*20)
        # This will hold {act_name: {metric_across_runs_mean: val, metric_across_runs_std: val}}
        combined_aggregated_val_results = {}
        for i in range(1, args.total_jobs + 1):
            part_suffix = f"_part{i}"
            val_summary_pkl_path = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_aggregated_validation_summary{part_suffix}.pkl")
            if os.path.exists(val_summary_pkl_path):
                print(f"Loading partial validation summary from: {val_summary_pkl_path}")
                try:
                    with open(val_summary_pkl_path, 'rb') as f_pkl:
                        partial_val_summary = pickle.load(f_pkl)
                    # The structure is {act_name: {metric_name_mean: val, metric_name_std: val}}
                    # Since activations are disjoint between parts, a simple update works.
                    combined_aggregated_val_results.update(partial_val_summary)
                except Exception as e:
                    print(f"Warning: Could not load or process {val_summary_pkl_path}: {e}")
            else:
                print(f"Warning: Partial validation summary file not found: {val_summary_pkl_path}")

        if combined_aggregated_val_results:
            metrics_from_val_summary_main_combine = ["psnr_mean", "ssim_mean", "epi_mean", "hf_recon_mean", "lpips_mean", "final_train_loss", "final_val_loss", "inf_time_ms"]
            print(f"\n--- COMBINED Aggregated Validation Table ({MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} - {NUM_RUNS} Run(s)) ---")
            header_val_agg = "Activation        " + "|".join([f"{m.replace('_mean','').replace('final_','').replace('_loss',' L').replace('hf_recon','HF-R').replace('inf_time_ms','Inf(ms)').center(12)}" for m in metrics_from_val_summary_main_combine if '_std' not in m])
            print(header_val_agg); print("-" * len(header_val_agg))

            # Sort by one of the metrics, e.g., final_val_loss_across_runs_mean
            # Ensure ALL_ACTIVATIONS_CONFIG.keys() is used for sorting to get the correct order of all activations
            sort_key_agg_val = 'final_val_loss_across_runs_mean'
            ranked_activations_val_agg = sorted(
                ALL_ACTIVATIONS_CONFIG.keys(), # Use all original keys for full sorted list
                key=lambda x: combined_aggregated_val_results.get(x, {}).get(sort_key_agg_val, float('inf'))
            )

            for act_name_print_val in ranked_activations_val_agg:
                 res_val_print = combined_aggregated_val_results.get(act_name_print_val, {})
                 line_val_print = f"{act_name_print_val:<18}"
                 for metric_print_val in metrics_from_val_summary_main_combine:
                     if '_std' in metric_print_val: continue # Only print means in this table
                     mean_val_print = res_val_print.get(f"{metric_print_val}_across_runs_mean", float('nan'))
                     # std_val_print = res_val_print.get(f"{metric_print_val}_across_runs_std", float('nan')) # if you want to print std too
                     if not np.isnan(mean_val_print): line_val_print += f"|{mean_val_print:^12.3f}"
                     else: line_val_print += f"|{'nan':^12}"
                 print(line_val_print)
            print("-" * len(header_val_agg))
            # Save combined validation summary
            combined_val_summary_path = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_COMBINED_aggregated_validation_summary.pkl")
            with open(combined_val_summary_path, 'wb') as f_out:
                pickle.dump(combined_aggregated_val_results, f_out)
            print(f"Saved combined aggregated validation summary to: {combined_val_summary_path}")
        else:
            print("No partial validation summary files found to combine.")

        print("\n" + "="*20 + f" {MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} Combination Complete " + "="*20)
        # End of --combine_results block. Script will exit after this.

    # --- Training and evaluation logic (if not --combine_results) ---
    else:
        if torch.cuda.is_available(): device = torch.device("cuda"); torch.backends.cudnn.benchmark = True; print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else: device = torch.device("cpu"); print("Using CPU")
        print(f"Selected Base Model: {MODEL_CHOICE}")
        print(f"Experiment data root: {DATASET_ROOT}"); print(f"Model save directory: {MODEL_DIR}"); print(f"Results output directory: {RESULT_DIR}")

        norm_mean = [0.5] * N_CHANNELS; norm_std = [0.5] * N_CHANNELS # Normalize to [-1, 1]
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Normalize(mean=norm_mean, std=norm_std)])
        test_transform = transforms.Compose([transforms.Normalize(mean=norm_mean, std=norm_std)])
        print(f"\nAttempting to load {DATASET_NAME}/{TASK_NAME} dataset (tensors) from: {DATASET_ROOT}")
        try:
            train_dataset_main = MedicalImageRestorationDataset(DATASET_ROOT, 'train', TASK_NAME, DEGRADED_FOLDER_NAME, train_transform, TARGET_SIZE, N_CHANNELS)
            val_dataset_main = MedicalImageRestorationDataset(DATASET_ROOT, 'val', TASK_NAME, DEGRADED_FOLDER_NAME, test_transform, TARGET_SIZE, N_CHANNELS)
            test_dataset_main = MedicalImageRestorationDataset(DATASET_ROOT, 'test', TASK_NAME, DEGRADED_FOLDER_NAME, test_transform, TARGET_SIZE, N_CHANNELS)
            if len(train_dataset_main) == 0 or len(val_dataset_main) == 0 or len(test_dataset_main) == 0: raise ValueError("One or more dataset splits are empty.")
            print(f"Loaded datasets: {len(train_dataset_main)} train, {len(val_dataset_main)} val, {len(test_dataset_main)} test tensors.")

            hpo_train_loader_main = DataLoader(train_dataset_main, BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True) if HPO_ENABLED else None
            hpo_val_loader_main = DataLoader(val_dataset_main, BATCH_SIZE, shuffle=False, num_workers=1) if HPO_ENABLED else None
        except Exception as e_load: print(f"\nFATAL ERROR loading dataset: {e_load}"); traceback.print_exc(); exit(1)

        num_workers_main = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1)
        pin_memory_flag_main = torch.cuda.is_available()
        # persistent_workers can cause issues on some systems, especially Windows, or with small datasets
        persist_workers_flag_main = (num_workers_main > 0 and os.name != 'nt' and len(train_dataset_main) > BATCH_SIZE * num_workers_main)

        train_loader_main_loop = DataLoader(train_dataset_main, BATCH_SIZE, shuffle=True, num_workers=num_workers_main, pin_memory=pin_memory_flag_main, persistent_workers=persist_workers_flag_main, drop_last=True)
        val_loader_main_loop = DataLoader(val_dataset_main, BATCH_SIZE, shuffle=False, num_workers=num_workers_main, pin_memory=pin_memory_flag_main, persistent_workers=persist_workers_flag_main)
        final_test_loader_for_anova = DataLoader(test_dataset_main, BATCH_SIZE, shuffle=False, num_workers=num_workers_main, pin_memory=pin_memory_flag_main, persistent_workers=persist_workers_flag_main)

        all_run_aggregated_val_results_main = {run: {} for run in range(NUM_RUNS)} # For current job part
        all_test_image_results_for_anova_this_part = [] # For current job part
        metrics_from_val_summary_main = ["psnr_mean", "ssim_mean", "epi_mean", "hf_recon_mean", "lpips_mean", "final_train_loss", "final_val_loss", "inf_time_ms"]
        experiment_start_time_main = time.time()
        per_image_csv_path_this_part = "N/A" # Path for this part's CSV

        for run_idx_main_loop in range(NUM_RUNS):
            job_part_info = f"Part {args.job_part}/{args.total_jobs}" if args.job_part is not None else "Full"
            print(f"\n{'='*20} STARTING {MODEL_CHOICE} RUN {run_idx_main_loop+1}/{NUM_RUNS} ({job_part_info}) on {DATASET_NAME}/{TASK_NAME} {'='*20}")
            current_seed = run_idx_main_loop # Or a more sophisticated seed strategy
            torch.manual_seed(current_seed); np.random.seed(current_seed); random.seed(current_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)

            for act_name_main, act_config_main in activations_to_run_this_job.items():
                print(f"\n--- Starting Activation: {act_name_main} (Run {run_idx_main_loop+1}) ---")
                act_start_time_main = time.time(); best_lr_main = DEFAULT_LR
                if HPO_ENABLED:
                    if hpo_train_loader_main and hpo_val_loader_main:
                        print(f"--- Optimizing LR for {act_name_main} via Optuna ---")
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                        study_hpo = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, HPO_N_EPOCHS // 3)))
                        objective_func_hpo = lambda trial: objective(trial, act_name_main, act_config_main, hpo_train_loader_main, hpo_val_loader_main, device)
                        try:
                            study_hpo.optimize(objective_func_hpo, n_trials=HPO_N_TRIALS, timeout=600) # Timeout to prevent excessive HPO time
                            if study_hpo.best_trial: best_lr_main = study_hpo.best_params["lr"]; print(f"--- Optuna Best LR for {act_name_main}: {best_lr_main:.2e} (Val Loss: {study_hpo.best_value:.6f}) ---")
                            else: print(f"--- Optuna no valid trials for {act_name_main}. Using default LR: {DEFAULT_LR:.2e} ---")
                        except Exception as e_hpo_opt: print(f"--- Optuna failed for {act_name_main}: {e_hpo_opt}. Using default LR. ---")
                        optuna.logging.set_verbosity(optuna.logging.INFO) # Restore verbosity
                    else: print(f"--- Skipping HPO for {act_name_main}: Loaders not available. ---")
                else: print(f"--- HPO disabled. Using default LR for {act_name_main}: {DEFAULT_LR:.2e} ---")

                current_wd_main = GLOBAL_WEIGHT_DECAY
                try:
                    aggregated_val_summary_run_act = train_and_evaluate(act_name_main, act_config_main, best_lr_main, current_wd_main, train_loader_main_loop, val_loader_main_loop, run_idx_main_loop, device, RESULT_DIR, MODEL_DIR)
                    all_run_aggregated_val_results_main[run_idx_main_loop][act_name_main] = aggregated_val_summary_run_act

                    # Evaluate on test set for this activation and run
                    best_model_path_for_test = os.path.join(MODEL_DIR, f"{MODEL_CHOICE.lower()}_{DATASET_NAME.lower()}_{TASK_NAME}_{act_name_main}_run{run_idx_main_loop+1}_best.pth")
                    if os.path.exists(best_model_path_for_test):
                        activation_instance_for_test = copy.deepcopy(act_config_main["fn"]) # Get a fresh instance
                        per_image_test_metrics_current_act_run = evaluate_model_on_test_set(best_model_path_for_test, activation_instance_for_test, final_test_loader_for_anova, device, act_name_main, run_idx_main_loop)
                        all_test_image_results_for_anova_this_part.extend(per_image_test_metrics_current_act_run)
                    else: print(f"WARNING: Best model not found for {act_name_main} Run {run_idx_main_loop+1} at {best_model_path_for_test}. Skipping test eval.")
                except Exception as e_train_eval:
                     print(f"\n!!! ERROR during main process for {act_name_main} Run {run_idx_main_loop+1}: {e_train_eval}"); traceback.print_exc()
                     # Add placeholder for failed run/activation in summary
                     all_run_aggregated_val_results_main[run_idx_main_loop][act_name_main] = {metric: float('nan') for metric in metrics_from_val_summary_main}
                act_end_time_main = time.time(); print(f"--- Finished Activation: {act_name_main} (Run {run_idx_main_loop+1}) - Time: {act_end_time_main - act_start_time_main:.2f} sec ---")
            print(f"========== COMPLETED {MODEL_CHOICE} RUN {run_idx_main_loop+1}/{NUM_RUNS} ({job_part_info}) ==========")

        # --- Process and save results for THIS JOB PART ---
        job_part_suffix = f"_part{args.job_part}" if args.job_part is not None else ""

        # 1. Aggregated validation results for this part
        aggregated_val_results_across_runs_this_part = {} # Key: act_name, Value: {metric_mean: val, metric_std: val}
        for act_name_agg_val in activations_to_run_this_job.keys(): # Only for activations processed by this job
            aggregated_val_results_across_runs_this_part[act_name_agg_val] = {}
            for metric_key_agg_val in metrics_from_val_summary_main:
                values_across_runs = [all_run_aggregated_val_results_main[run_idx_agg].get(act_name_agg_val, {}).get(metric_key_agg_val, float('nan')) for run_idx_agg in range(NUM_RUNS)]
                valid_values_agg = [v for v in values_across_runs if v is not None and np.isfinite(v)]
                mean_agg = np.mean(valid_values_agg) if valid_values_agg else float('nan')
                std_agg = np.std(valid_values_agg) if len(valid_values_agg) > 1 else 0.0
                aggregated_val_results_across_runs_this_part[act_name_agg_val][f"{metric_key_agg_val}_across_runs_mean"] = mean_agg
                aggregated_val_results_across_runs_this_part[act_name_agg_val][f"{metric_key_agg_val}_across_runs_std"] = std_agg if not np.isnan(mean_agg) else float('nan')

        # Save this part's aggregated validation summary
        val_summary_pkl_path_this_part = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_aggregated_validation_summary{job_part_suffix}.pkl")
        try:
            with open(val_summary_pkl_path_this_part, 'wb') as f_pkl:
                pickle.dump(aggregated_val_results_across_runs_this_part, f_pkl)
            print(f"Saved aggregated validation summary for this job part to: {val_summary_pkl_path_this_part}")
        except Exception as e_pkl_save:
            print(f"Error saving validation summary pkl for this part: {e_pkl_save}")

        # Print validation summary table for THIS PART
        print(f"\n--- Aggregated Validation Table for Job Part ({job_part_info}, {MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} - {NUM_RUNS} Run(s)) ---")
        header_val_agg_part = "Activation        " + "|".join([f"{m.replace('_mean','').replace('final_','').replace('_loss',' L').replace('hf_recon','HF-R').replace('inf_time_ms','Inf(ms)').center(12)}" for m in metrics_from_val_summary_main if '_std' not in m])
        print(header_val_agg_part); print("-" * len(header_val_agg_part))
        sort_key_agg_val_part = 'final_val_loss_across_runs_mean'
        ranked_activations_val_agg_part = sorted(
            activations_to_run_this_job.keys(),
            key=lambda x: aggregated_val_results_across_runs_this_part.get(x, {}).get(sort_key_agg_val_part, float('inf'))
        )
        for act_name_print_val_part in ranked_activations_val_agg_part:
             res_val_print_part = aggregated_val_results_across_runs_this_part.get(act_name_print_val_part, {})
             line_val_print_part = f"{act_name_print_val_part:<18}"
             for metric_print_val_part in metrics_from_val_summary_main:
                 if '_std' in metric_print_val_part: continue
                 mean_val_print_part = res_val_print_part.get(f"{metric_print_val_part}_across_runs_mean", float('nan'))
                 if not np.isnan(mean_val_print_part): line_val_print_part += f"|{mean_val_print_part:^12.3f}"
                 else: line_val_print_part += f"|{'nan':^12}"
             print(line_val_print_part)
        print("-" * len(header_val_agg_part))


        # 2. Per-Image test metrics for this part (for ANOVA later)
        if all_test_image_results_for_anova_this_part:
            print(f"\n--- Saving Per-Image Test Set Metrics for this Job Part ({job_part_info}) ---")
            per_image_df_for_anova_part = pd.DataFrame(all_test_image_results_for_anova_this_part)
            per_image_csv_path_this_part = os.path.join(RESULT_DIR, "metrics", f"{EXPERIMENT_FOLDER_NAME}_per_image_test_metrics{job_part_suffix}.csv")
            try:
                per_image_df_for_anova_part.to_csv(per_image_csv_path_this_part, index=False, float_format='%.6f')
                print(f"Per-image test metrics for this job part saved to: {per_image_csv_path_this_part}")
                # ANOVA is NOT run here for partial jobs. It's run in --combine_results mode.
                if args.job_part is None: # If it was a full run (not a part), run ANOVA now
                    print("\n--- Performing ANOVA Analysis on Full Run Test Set Metrics ---")
                    metrics_for_anova_final = ["PSNR", "SSIM", "EPI", "HF_Score", "LPIPS"]
                    metrics_present_for_anova = [m for m in metrics_for_anova_final if m in per_image_df_for_anova_part.columns]
                    if metrics_present_for_anova:
                        run_anova_analysis(per_image_df_for_anova_part, metrics_present_for_anova, RESULT_DIR)
                    else: print("No valid metrics found in DataFrame for ANOVA analysis.")
            except IOError as e_csv_io: print(f"\nError saving per-image results CSV for this part: {e_csv_io}")
            except Exception as e_part_anova: print(f"\nError during ANOVA for this part: {e_part_anova}"); traceback.print_exc()
        else:
            print("\nNo per-image test results collected for this job part. Skipping CSV saving.")

        experiment_end_time_main = time.time(); total_time_main = experiment_end_time_main - experiment_start_time_main
        print("\n" + "="*20 + f" {MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} Analysis for Job Part ({job_part_info}) Complete " + "="*20)
        print(f"Total Time for this Job Part: {total_time_main / 60:.2f} minutes ({total_time_main:.2f} seconds)")
        print(f"Results for this part saved in base directories: {os.path.abspath(RESULT_DIR)} and {os.path.abspath(MODEL_DIR)}")
        if all_test_image_results_for_anova_this_part:
            print(f"Per-image test metrics for ANOVA (this part) saved to: {os.path.abspath(per_image_csv_path_this_part)}")
        print("=" * (60 + len(f" {MODEL_CHOICE}/{DATASET_NAME}/{TASK_NAME} Analysis for Job Part ({job_part_info}) Complete.... ")))