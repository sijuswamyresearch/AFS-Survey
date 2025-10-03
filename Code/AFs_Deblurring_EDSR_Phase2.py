# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import convolve
import random
import math
import warnings
import time
import copy # For deep copying model state
import traceback # For printing tracebacks
from collections import defaultdict
from functools import partial # For hooks

# --- Imports for HPO, Metrics ---
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. Running without Hyperparameter Optimization. Using default LR.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS library not found. LPIPS metric will not be calculated.")

# REMOVED thop import
THOP_AVAILABLE = False # Force disable thop features

# --- Custom Activation Functions ---
class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.sigmoid(x)

class ESwish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__(); self.beta = nn.Parameter(torch.tensor(beta))
    def forward(self, x): return self.beta * x * torch.sigmoid(x)

class Aria(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__(); self.alpha = nn.Parameter(torch.tensor(alpha)); self.beta = nn.Parameter(torch.tensor(beta))
    def forward(self, x):
        x = torch.clamp(x, -20, 20); exp_term = torch.exp(-self.alpha * x); sin_term = torch.sin(self.beta * x)
        return exp_term * sin_term

class GCU(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x * torch.cos(torch.clamp(x, -math.pi, math.pi))

class Snake(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__(); self.alpha = nn.Parameter(torch.tensor(alpha))
    def forward(self, x):
        alpha_safe = torch.where(self.alpha == 0, torch.tensor(1e-7, device=self.alpha.device, dtype=self.alpha.dtype), self.alpha)
        return x + (torch.sin(alpha_safe * x) ** 2 / alpha_safe)

class FReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0: raise ValueError(f"FReLU requires positive integer channels, got {channels}")
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1); nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if x.dim() != 4: raise ValueError(f"FReLU expects 4D input (NCHW), got {x.dim()}D")
        spatial_cond = self.bn(self.conv(x))
        out = torch.max(x, spatial_cond)
        return out

# --- Configuration ---
DATA_ROOT = "deblur_dataset"
SYNTHETIC_DIR = os.path.join(DATA_ROOT, "synthetic")
MODEL_DIR = "models_edsr" # EDSR specific
RESULT_DIR_BASE = "results"
RESULT_DIR = os.path.join(RESULT_DIR_BASE, "edsr") # EDSR specific

os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "spatial_analysis"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "fft_analysis"), exist_ok=True)

# --- Hyperparameters ---
IMG_PATH = "/dist_home/siju/AFS/Sample1.jpg" # change this path
BATCH_SIZE = 8
EPOCHS = 100
TARGET_SIZE = (256, 256)
NUM_SAMPLES = 100
TEST_SPLIT = 0.2
NUM_RUNS = 3
GLOBAL_WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# --- EDSR Hyperparameters ---
N_RESBLOCKS = 16
N_FEATS = 64
RES_SCALE = 0.1

# --- HPO Configuration ---
HPO_ENABLED = OPTUNA_AVAILABLE
HPO_N_TRIALS = 15
HPO_N_EPOCHS = 5
DEFAULT_LR = 1e-4
LR_RANGE = [1e-5, 1e-3]

# --- Activation configurations ---
ACTIVATIONS = {
    "ReLU": {"fn": nn.ReLU()}, "LeakyReLU": {"fn": nn.LeakyReLU(0.1)},
    "Sigmoid": {"fn": nn.Sigmoid()}, "Tanh": {"fn": nn.Tanh()},
    "ELU": {"fn": nn.ELU(alpha=1.0)}, "SiLU": {"fn": nn.SiLU()},
    "Mish": {"fn": Mish()}, "Swish": {"fn": Swish()},
    "ESwish": {"fn": ESwish(beta=1.25)}, "Aria": {"fn": Aria(alpha=0.5, beta=1.0)},
    "GCU": {"fn": GCU()}, "Snake": {"fn": Snake(alpha=0.5)},
    "FReLU": {"fn": FReLU(channels=1)} # Placeholder instance
}
PARAMETRIC_ACTIVATIONS = (ESwish, Aria, Snake, FReLU, nn.PReLU)

# --- Advanced Blurring Techniques ---
class AdvancedBlur:
    @staticmethod
    def gaussian_blur(img_tensor, kernel_size=5, sigma=3.0):
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        blur = transforms.GaussianBlur(kernel_size, sigma=sigma)
        return blur(img_tensor)
    @staticmethod
    def motion_blur(img_tensor, kernel_size=15):
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        img_np = (img_np * 255).astype(np.uint8)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        blurred = np.zeros_like(img_np)
        for i in range(3): blurred[..., i] = convolve(img_np[..., i], kernel)
        blurred = torch.from_numpy(blurred.astype(np.float32)) / 255.0
        blurred = (blurred.permute(2, 0, 1) - 0.5) / 0.5
        return blurred.to(img_tensor.device)
    @staticmethod
    def defocus_blur(img_tensor, radius=3):
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        img_np = (img_np * 255).astype(np.uint8)
        kernel_size = radius * 2 + 1; blurred = cv2.blur(img_np, (kernel_size, kernel_size))
        if blurred.ndim == 2: blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        blurred = torch.from_numpy(blurred.astype(np.float32)) / 255.0
        blurred = (blurred.permute(2, 0, 1) - 0.5) / 0.5
        return blurred.to(img_tensor.device)
    @staticmethod
    def random_blur(img_tensor):
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        if blur_type == 'gaussian':
            sigma = random.uniform(1.0, 5.0); kernel_size = random.choice([5, 7, 9])
            return AdvancedBlur.gaussian_blur(img_tensor, kernel_size=kernel_size, sigma=sigma)
        elif blur_type == 'motion':
            kernel_size = random.choice([9, 15, 21])
            return AdvancedBlur.motion_blur(img_tensor, kernel_size=kernel_size)
        else:
            radius = random.randint(2, 5)
            return AdvancedBlur.defocus_blur(img_tensor, radius=radius)

# --- Dataset generation ---
def generate_synthetic_dataset(force_generate=False):
    if not force_generate and os.path.exists(SYNTHETIC_DIR) and os.listdir(SYNTHETIC_DIR):
         print(f"Dataset exists in {SYNTHETIC_DIR}. Skipping."); return
    elif force_generate and os.path.exists(SYNTHETIC_DIR):
         print(f"Forcing regeneration. Clearing {SYNTHETIC_DIR}...");
         for filename in os.listdir(SYNTHETIC_DIR):
            file_path = os.path.join(SYNTHETIC_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            except Exception as e: print(f'Failed to delete {file_path}. Reason: {e}')
    elif not os.path.exists(SYNTHETIC_DIR): os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    if not os.path.exists(IMG_PATH): print(f"ERROR: Source image not found: {IMG_PATH}"); return
    print("Generating synthetic dataset..."); transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img = Image.open(IMG_PATH).convert('RGB'); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in tqdm(range(NUM_SAMPLES), desc="Generating synthetic dataset"):
        img_transformed = transform(img).to(device); blurred = AdvancedBlur.random_blur(img_transformed)
        torch.save(blurred.cpu(), os.path.join(SYNTHETIC_DIR, f"blurred_{i:04d}.pt"))
        torch.save(img_transformed.cpu(), os.path.join(SYNTHETIC_DIR, f"sharp_{i:04d}.pt"))
    print("Dataset generation complete.")

# --- Dataset loader ---
class PreGeneratedDeblurDataset(Dataset):
    def __init__(self, root_dir, num_samples=None):
        self.root_dir = root_dir;
        if not os.path.isdir(root_dir): raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        all_blurred = sorted([f for f in os.listdir(root_dir) if f.startswith("blurred_") and f.endswith(".pt")])
        max_samples = len(all_blurred)
        num_samples_to_use = min(num_samples, max_samples) if num_samples else max_samples
        if num_samples_to_use == 0: raise FileNotFoundError(f"No blurred_*.pt files found in {root_dir}")
        self.samples = [(f"blurred_{i:04d}.pt", f"sharp_{i:04d}.pt") for i in range(num_samples_to_use)]
        print(f"Loading {len(self.samples)} samples from {root_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        blurred_path, sharp_path = self.samples[idx];
        try:
            blurred = torch.load(os.path.join(self.root_dir, blurred_path), weights_only=False)
            sharp = torch.load(os.path.join(self.root_dir, sharp_path), weights_only=False)
            return blurred, sharp
        except FileNotFoundError as e: print(f"Error loading sample {idx}: {e}"); raise e
        except Exception as e: print(f"Error loading files {blurred_path}, {sharp_path}: {e}"); raise e

# --- EDSR architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, n_feats, activation_fn=nn.ReLU(), res_scale=0.1):
        super(ResidualBlock, self).__init__()
        if isinstance(activation_fn, FReLU): self.activation = FReLU(channels=n_feats)
        elif isinstance(activation_fn, PARAMETRIC_ACTIVATIONS): self.activation = copy.deepcopy(activation_fn)
        else: self.activation = activation_fn
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True), self.activation,
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        )
        self.res_scale = res_scale
    def forward(self, x): return x + self.body(x).mul(self.res_scale)

class EDSRDeblur(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS,
                 res_scale=RES_SCALE, activation_fn=nn.ReLU()):
        super(EDSRDeblur, self).__init__()
        self.activation_fn_template = copy.deepcopy(activation_fn)
        self.head = nn.Conv2d(n_channels, n_feats, 3, 1, 1, bias=True)
        body_modules = [
            ResidualBlock(n_feats, activation_fn=self.activation_fn_template, res_scale=res_scale)
            for _ in range(n_resblocks) ]
        self.body = nn.Sequential(*body_modules)
        self.tail = nn.Conv2d(n_feats, n_classes, 3, 1, 1, bias=True)
        self.final_activation = nn.Tanh()
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        s = x; x = self.head(x); res = self.body(x); res = res + x
        x = self.tail(res); output = self.final_activation(x)
        return output

# --- Evaluation metrics ---
def calculate_epi(original: torch.Tensor, denoised: torch.Tensor, window_size: int = 5) -> float:
    original = original.detach().cpu(); denoised = denoised.detach().cpu()
    if original.dim() == 4: original = original.squeeze(0)
    if denoised.dim() == 4: denoised = denoised.squeeze(0)
    if original.shape[0] == 3: original_np = (0.299*original[0].numpy()+0.587*original[1].numpy()+0.114*original[2].numpy()); denoised_np = (0.299*denoised[0].numpy()+0.587*denoised[1].numpy()+0.114*denoised[2].numpy())
    else: original_np = original[0].numpy(); denoised_np = denoised[0].numpy()
    original_np = original_np.astype(np.float32); denoised_np = denoised_np.astype(np.float32)
    grad_x_orig = cv2.Sobel(original_np, cv2.CV_32F, 1, 0, ksize=3); grad_y_orig = cv2.Sobel(original_np, cv2.CV_32F, 0, 1, ksize=3); grad_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)
    grad_x_den = cv2.Sobel(denoised_np, cv2.CV_32F, 1, 0, ksize=3); grad_y_den = cv2.Sobel(denoised_np, cv2.CV_32F, 0, 1, ksize=3); grad_den = np.sqrt(grad_x_den**2 + grad_y_den**2)
    grad_orig = (grad_orig - grad_orig.min()) / (grad_orig.max() - grad_orig.min() + 1e-7); grad_den = (grad_den - grad_den.min()) / (grad_den.max() - grad_den.min() + 1e-7); pad = window_size // 2
    grad_orig_pad = np.pad(grad_orig, pad, mode='reflect'); grad_den_pad = np.pad(grad_den, pad, mode='reflect'); epi_values = []; rows, cols = grad_orig.shape
    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            window_orig = grad_orig_pad[i-pad:i+pad+1, j-pad:j+pad+1]; window_den = grad_den_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            mean_orig = np.mean(window_orig); mean_den = np.mean(window_den); numerator = np.sum((window_orig - mean_orig) * (window_den - mean_den))
            denom_orig_sq = np.sum((window_orig - mean_orig)**2); denom_den_sq = np.sum((window_den - mean_den)**2); denominator = np.sqrt(denom_orig_sq * denom_den_sq);
            if denominator > 1e-7: epi = np.clip(numerator / denominator, -1.0, 1.0); epi_values.append(epi)
    return np.mean(epi_values) if epi_values else 0.0

def calculate_hf_energy_ratio(original, denoised):
    def get_hf_energy(img):
        img = img.detach().cpu();
        if img.dim() == 4: img = img.squeeze(0)
        if img.shape[0] == 3: img_gray = 0.299*img[0]+0.587*img[1]+0.114*img[2]
        else: img_gray = img[0]
        fft = torch.fft.fft2(img_gray); fft_shift = torch.fft.fftshift(fft); h, w = img_gray.shape; cy, cx = h // 2, w // 2
        radius_ratio = 0.1; radius = radius_ratio * min(cx, cy)
        y, x = torch.meshgrid(torch.arange(h)-cy, torch.arange(w)-cx, indexing='ij'); mask = (x**2+y**2) > (radius**2)
        hf_energy = torch.sum(torch.abs(fft_shift) * mask); return hf_energy
    hf_original = get_hf_energy(original); hf_denoised = get_hf_energy(denoised)
    return (hf_denoised / (hf_original + 1e-9)).item()

lpips_model_global = None; lpips_model_failed = False
def calculate_all_metrics(original, denoised, device):
    global lpips_model_global, lpips_model_failed
    original_d = original.detach(); denoised_d = denoised.detach()
    original_b = original_d.unsqueeze(0).to(device); denoised_b = denoised_d.unsqueeze(0).to(device)
    original_np = (original_d.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    denoised_np = (denoised_d.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    try: psnr_val = psnr(original_np, denoised_np, data_range=1.0)
    except ValueError: psnr_val = float('nan')
    try:
        win_size = min(7, original_np.shape[0], original_np.shape[1]); win_size = max(3, win_size if win_size % 2 == 1 else win_size - 1)
        if original_np.ndim == 3 and original_np.shape[2] >= 3: ssim_val = ssim(original_np, denoised_np, data_range=1.0, channel_axis=2, win_size=win_size)
        elif original_np.ndim >= 2: ssim_val = ssim(original_np.squeeze(), denoised_np.squeeze(), data_range=1.0, win_size=win_size)
        else: ssim_val = float('nan')
    except ValueError: ssim_val = float('nan')
    epi_val = calculate_epi(original_d, denoised_d); hf_ratio_val = calculate_hf_energy_ratio(original_d, denoised_d)
    lpips_val = float('nan')
    if LPIPS_AVAILABLE and not lpips_model_failed:
        if lpips_model_global is None:
            try: print("Initializing LPIPS model..."); lpips_model_global = lpips.LPIPS(net='alex').to(device); lpips_model_global.eval()
            except Exception as e: print(f"Failed to initialize LPIPS: {e}"); lpips_model_failed = True
        if not lpips_model_failed:
            with torch.no_grad():
                try: lpips_val = lpips_model_global(original_b, denoised_b).item()
                except Exception as e: print(f"LPIPS calculation error: {e}")
    return {"psnr": psnr_val, "ssim": ssim_val, "epi": epi_val, "hf_recon": hf_ratio_val, "lpips": lpips_val}

# --- Activation analyzer class ---
class ActivationAnalyzer:
    def __init__(self, model):
        self.model = model; self.pre_act_maps = defaultdict(list); self.post_act_maps = defaultdict(list); self.hooks = []
    def _pre_act_hook(self, name, module, inp):
        if isinstance(inp, tuple): inp = inp[0]
        if inp is not None and isinstance(inp, torch.Tensor) and inp.nelement() > 0: self.pre_act_maps[name].append(inp[0].detach().cpu())
    def _post_act_hook(self, name, module, inp, out):
        if isinstance(out, tuple): out = out[0]
        if out is not None and isinstance(out, torch.Tensor) and out.nelement() > 0: self.post_act_maps[name].append(out[0].detach().cpu())
    def register_hooks(self):
        self.remove_hooks(); self.pre_act_maps.clear(); self.post_act_maps.clear()
        hook_count = 0
        target_act_types = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.SiLU, Mish, Swish, ESwish, Aria, GCU, Snake, FReLU)
        for name, module in self.model.named_modules():
            if isinstance(module, target_act_types):
                self.hooks.append(module.register_forward_pre_hook(partial(self._pre_act_hook, name)))
                self.hooks.append(module.register_forward_hook(partial(self._post_act_hook, name)))
                hook_count += 2
    def remove_hooks(self):
        for hook in self.hooks: hook.remove(); self.hooks = []
    def analyze_batch(self, input_batch):
        if input_batch.nelement() == 0: print("Warn: Empty input batch to Analyzer."); return
        self.model.eval();
        with torch.no_grad(): _ = self.model(input_batch.to(next(self.model.parameters()).device))
    def get_activation_maps(self, layer_name):
        pre = self.pre_act_maps.get(layer_name, [None])[0]; post = self.post_act_maps.get(layer_name, [None])[0]
        return pre, post

# --- Analysis functions ---
def save_feature_maps(pre_act, post_act, act_name, layer_name, channel=0, result_dir=RESULT_DIR):
    if pre_act is None or post_act is None: return
    if pre_act.dim() == 2: pre_act = pre_act.unsqueeze(0)
    if post_act.dim() == 2: post_act = post_act.unsqueeze(0)
    if pre_act.dim() == 3: pre_act = pre_act.unsqueeze(0)
    if post_act.dim() == 3: post_act = post_act.unsqueeze(0)
    if pre_act.numel()==0 or post_act.numel()==0 or pre_act.shape[1]==0 or post_act.shape[1]==0: return
    channel = min(channel, pre_act.shape[1]-1, post_act.shape[1]-1);
    if channel < 0: return
    if pre_act.shape[-2:] != post_act.shape[-2:]:
        target_size = (max(pre_act.shape[-2],post_act.shape[-2]), max(pre_act.shape[-1],post_act.shape[-1]))
        pre_act = F.interpolate(pre_act, size=target_size, mode='bilinear', align_corners=False)
        post_act = F.interpolate(post_act, size=target_size, mode='bilinear', align_corners=False)
    pre = pre_act[0, channel].numpy(); post = post_act[0, channel].numpy(); diff = post - pre
    plt.figure(figsize=(15, 5)); plt.clf()
    plt.subplot(131); im1=plt.imshow(pre, cmap='viridis'); plt.title(f"Pre-Act ({layer_name}, C{channel})"); plt.colorbar(im1); plt.axis('off')
    plt.subplot(132); im2=plt.imshow(post, cmap='viridis'); plt.title(f"Post-Act ({act_name})"); plt.colorbar(im2); plt.axis('off')
    plt.subplot(133); im3=plt.imshow(diff, cmap='coolwarm', vmin=np.percentile(diff,1), vmax=np.percentile(diff,99)); plt.title("Difference (Post-Pre)"); plt.colorbar(im3); plt.axis('off')
    save_dir = os.path.join(result_dir, "spatial_analysis"); os.makedirs(save_dir, exist_ok=True)
    layer_name_safe = layer_name.replace('.','_').replace(':','_')
    save_path = os.path.join(save_dir, f"edsr_{act_name}_{layer_name_safe}_C{channel}.png") # edsr prefix
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save feature map {save_path}: {e}")
    finally: plt.close()

def save_comparison_plot(blurred, restored, original, act_name, run_idx, metrics=None, result_dir=RESULT_DIR):
    blurred=blurred.detach().cpu(); restored=restored.detach().cpu(); original=original.detach().cpu()
    def prepare_img(img):
        if img.dim()==4: img=img.squeeze(0)
        img = img.permute(1, 2, 0).numpy()*0.5+0.5; return np.clip(img, 0, 1)
    metric_str = ""
    if metrics:
        items = []
        if 'psnr' in metrics and not np.isnan(metrics['psnr']): items.append(f"PSNR: {metrics['psnr']:.2f} dB")
        if 'ssim' in metrics and not np.isnan(metrics['ssim']): items.append(f"SSIM: {metrics['ssim']:.3f}")
        if 'lpips' in metrics and not np.isnan(metrics['lpips']): items.append(f"LPIPS: {metrics['lpips']:.3f}")
        if 'epi' in metrics and not np.isnan(metrics['epi']): items.append(f"EPI: {metrics['epi']:.3f}")
        if 'hf_recon' in metrics and not np.isnan(metrics['hf_recon']): items.append(f"HF: {metrics['hf_recon']*100:.1f}%")
        metric_str = "\n" + " | ".join(items)
    plt.figure(figsize=(18, 6)); plt.clf(); plt.suptitle(f"EDSR Run {run_idx+1}", fontsize=16) # EDSR title
    plt.subplot(131); plt.imshow(prepare_img(blurred)); plt.title("Blurred Input"); plt.axis('off')
    plt.subplot(132); plt.imshow(prepare_img(restored)); plt.title(f"{act_name} Restored" + metric_str); plt.axis('off')
    plt.subplot(133); plt.imshow(prepare_img(original)); plt.title("Ground Truth"); plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_dir = result_dir; os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"edsr_{act_name}_run{run_idx+1}_comparison.png") # edsr prefix
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save comparison plot {save_path}: {e}")
    finally: plt.close()

def plot_loss_curves(train_losses, val_losses, act_name, run_idx, result_dir=RESULT_DIR):
    epochs = range(1, len(train_losses) + 1); plt.figure(figsize=(10, 6)); plt.clf()
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'EDSR Loss Curves ({act_name} - Run {run_idx+1})', fontsize=14) # EDSR title
    plt.xlabel('Epoch', fontsize=12); plt.ylabel('Loss (L1)', fontsize=12)
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3); plt.ylim(bottom=0)
    save_dir = os.path.join(result_dir, "metrics"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"edsr_{act_name}_run{run_idx+1}_loss_curves.png") # edsr prefix
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save loss curve {save_path}: {e}")
    finally: plt.close()

def analyze_frequency_and_spatial(model, analyzer, gt_img, pred_img, blurred_img, act_name, run_idx, result_dir=RESULT_DIR):
    device = next(model.parameters()).device
    gt_img_cpu=gt_img.detach().cpu(); pred_img_cpu=pred_img.detach().cpu(); blurred_img_cpu=blurred_img.detach().cpu()

    # --- FFT analysis part ---
    def prepare_spectrum(img):
        if img.dim() == 4: img = img.squeeze(0)
        if img.shape[0] == 3: gray = 0.299*img[0]+0.587*img[1]+0.114*img[2]
        else: gray = img[0]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9); return gray
    def compute_spectrum(img):
        fft=torch.fft.fft2(img.float()); shift=torch.fft.fftshift(fft)
        return torch.log(torch.abs(shift) + 1e-9)

    print(f"Analyzing frequency domain for {act_name} Run {run_idx+1}...")
    gt_spectrum=compute_spectrum(prepare_spectrum(gt_img_cpu)); blurred_spectrum=compute_spectrum(prepare_spectrum(blurred_img_cpu)); pred_spectrum=compute_spectrum(prepare_spectrum(pred_img_cpu))
    plt.figure(figsize=(18, 6)); plt.clf(); plt.suptitle(f"EDSR Run {run_idx+1}", fontsize=16) # EDSR title
    plt.subplot(131); plt.imshow(gt_spectrum, cmap='viridis'); plt.title("GT Spectrum"); plt.axis('off')
    plt.subplot(132); plt.imshow(blurred_spectrum, cmap='viridis'); plt.title("Blurred Spectrum"); plt.axis('off')
    plt.subplot(133); plt.imshow(pred_spectrum, cmap='viridis'); plt.title(f"{act_name} Spectrum"); plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_dir = os.path.join(result_dir, "fft_analysis"); os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"edsr_{act_name}_run{run_idx+1}_main_spectra.png") # edsr prefix
    try: plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e: print(f" Failed to save spectra plot {save_path}: {e}")
    finally: plt.close()

    # --- Spatial analysis part (using Analyzer) ---
    print(f"Analyzing spatial domain for {act_name} Run {run_idx+1}...")
    analyzer.register_hooks()
    analyzer.analyze_batch(blurred_img_cpu.unsqueeze(0).to(device))
    analyzer.remove_hooks()

    layers_to_visualize = list(analyzer.pre_act_maps.keys())
    if len(layers_to_visualize) > 10: layers_to_visualize = random.sample(layers_to_visualize, 10)

    for layer_name in layers_to_visualize:
        pre_map, post_map = analyzer.get_activation_maps(layer_name) # These are [C, H, W]
        if pre_map is not None and post_map is not None:
             num_channels = pre_map.shape[0]
             for channel in range(min(1, num_channels)):
                 layer_id_str = f"{layer_name}_run{run_idx+1}"
                 save_feature_maps(pre_map[channel], post_map[channel], # Pass C, H, W
                                   act_name, layer_id_str, 0, result_dir)

# --- HPO objective function ---
def objective(trial, act_name, act_config, hpo_train_loader, hpo_val_loader, device):
    lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    weight_decay = GLOBAL_WEIGHT_DECAY
    # Use EDSRDeblur here
    model = EDSRDeblur(activation_fn=copy.deepcopy(act_config["fn"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    model.train()
    for epoch in range(HPO_N_EPOCHS):
        num_batch_train = 0
        for blurred, sharp in hpo_train_loader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad(); output = model(blurred); loss = criterion(output, sharp)
            if torch.isnan(loss): print(f"NaN loss HPO trial {trial.number} {act_name}. Pruning."); raise optuna.TrialPruned()
            loss.backward(); optimizer.step(); num_batch_train+=1
            if num_batch_train >= 10: break
    model.eval(); val_loss = 0; num_batches=0
    with torch.no_grad():
        for blurred, sharp in hpo_val_loader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred); v_loss = criterion(output, sharp).item()
            if not np.isnan(v_loss): val_loss += v_loss; num_batches += 1
            if num_batches >= 5: break
    avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
    if np.isnan(avg_val_loss) or avg_val_loss == float('inf'): print(f"NaN/Inf val_loss HPO trial {trial.number} {act_name}. Pruning."); raise optuna.TrialPruned()
    trial.report(avg_val_loss, HPO_N_EPOCHS - 1)
    if trial.should_prune(): raise optuna.TrialPruned()
    return avg_val_loss

# --- Training & evaluation function (No thop) ---
def train_and_evaluate(act_name, act_config, learning_rate, weight_decay,
                       train_loader, test_loader, run_idx, device, result_dir=RESULT_DIR):

    print(f"\n=== Training EDSR Run {run_idx+1} with {act_name} (LR={learning_rate:.2e}) ===")
    # Use EDSRDeblur
    model = EDSRDeblur(activation_fn=copy.deepcopy(act_config["fn"])).to(device)

    analyzer = ActivationAnalyzer(model) # Analyzer for post-training analysis

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)
    criterion = nn.L1Loss()

    train_loss_history = []; val_loss_history = []; best_val_loss = float('inf')
    epochs_no_improve = 0; best_model_state = None

    # --- Training loop ---
    for epoch in range(EPOCHS):
        model.train(); epoch_loss = 0; num_train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for blurred, sharp in pbar:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad(); output = model(blurred); loss = criterion(output, sharp)
            if torch.isnan(loss): print(f"NaN train loss epoch {epoch+1}, skip batch."); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); epoch_loss += loss.item(); num_train_batches += 1; pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = epoch_loss / num_train_batches if num_train_batches > 0 else float('nan')
        train_loss_history.append(avg_train_loss)

        # --- Validation loop ---
        model.eval(); val_loss = 0; num_val_batches = 0
        pbar_val = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for blurred, sharp in pbar_val:
                blurred, sharp = blurred.to(device), sharp.to(device)
                output = model(blurred); v_loss = criterion(output, sharp).item()
                if not np.isnan(v_loss): val_loss += v_loss; num_val_batches += 1
                pbar_val.set_postfix(loss=f"{v_loss:.4f}")
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        current_val_loss_for_history = avg_val_loss if avg_val_loss != float('inf') else (val_loss_history[-1] if val_loss_history and not np.isnan(val_loss_history[-1]) else float('nan'))
        val_loss_history.append(current_val_loss_for_history)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f} (LR: {optimizer.param_groups[0]['lr']:.2e})")

        if avg_val_loss != float('inf') and not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; best_model_state = copy.deepcopy(model.state_dict()); epochs_no_improve = 0
            else: epochs_no_improve += 1
        else: epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(f"Early stopping @ epoch {epoch + 1}."); break

    print("Training finished.")
    plot_loss_curves(train_loss_history, val_loss_history, act_name, run_idx, result_dir=result_dir)

    if best_model_state is None: print("Warn: Using final model state."); best_model_state = model.state_dict()
    model.load_state_dict(best_model_state); model.eval()

    
    flops = float('nan'); params_thop = float('nan'); inference_time_ms = float('nan')

    # --- Inference time measurement ---
    try:
        num_inf_samples = min(len(test_loader.dataset), 32); inference_times = []
        with torch.no_grad():
             for _ in range(3): dummy_input, _ = next(iter(test_loader)); _ = model(dummy_input.to(device)); torch.cuda.synchronize() if device.type=='cuda' else None
             count=0
             for blurred, sharp in test_loader:
                 if count >= num_inf_samples: break; start_time = time.time()
                 _ = model(blurred.to(device)); torch.cuda.synchronize() if device.type=='cuda' else None
                 end_time = time.time(); inference_times.append((end_time - start_time) * 1000 / blurred.size(0))
                 count += blurred.size(0)
        if inference_times: inference_time_ms = np.mean(inference_times); print(f"Avg Inference Time: {inference_time_ms:.2f} ms/image")
    except Exception as e: print(f"Warn: Inference time measurement failed for {act_name}: {e}")

    # --- Metric calculation & visualization ---
    final_metrics = {"psnr": [], "ssim": [], "epi": [], "hf_recon": [], "lpips": []}
    num_eval_samples = min(len(test_loader.dataset), BATCH_SIZE * 4)
    eval_count = 0; vis_sample_collected = False; vis_blurred, vis_sharp, vis_output = None, None, None
    vis_indices = list(range(min(BATCH_SIZE, len(test_dataset))))
    vis_loader = DataLoader(Subset(test_dataset, vis_indices), batch_size=BATCH_SIZE)

    with torch.no_grad():
        try:
            vis_batch_blurred, vis_batch_sharp = next(iter(vis_loader))
            if vis_batch_blurred.nelement() > 0:
                 # IMPORTANT: Ensure model forward pass doesn't trigger thop hooks now
                 vis_output_dev = model(vis_batch_blurred.to(device))
                 vis_blurred = vis_batch_blurred[0].cpu()
                 vis_sharp = vis_batch_sharp[0].cpu()
                 vis_output = vis_output_dev[0].cpu()
                 vis_sample_collected = True
            else: print("Warn: Visualization batch is empty.")
        except StopIteration: print("Warn: Could not get visualization batch.")


        for blurred, sharp in test_loader:
            if eval_count >= num_eval_samples: break
            blurred_dev, sharp_dev = blurred.to(device), sharp.to(device)
            output_dev = model(blurred_dev)
            for j in range(blurred.size(0)):
                if eval_count >= num_eval_samples: break
                metrics = calculate_all_metrics(sharp_dev[j], output_dev[j], device)
                for key in final_metrics: final_metrics[key].append(metrics.get(key, float('nan')))
                eval_count += 1

    avg_final_metrics = {}
    for key, values in final_metrics.items():
         valid_values = [v for v in values if not np.isnan(v)]
         avg_final_metrics[key] = np.mean(valid_values) if valid_values else float('nan')
    avg_final_metrics['final_train_loss'] = train_loss_history[-1] if train_loss_history and not np.isnan(train_loss_history[-1]) else float('nan')
    avg_final_metrics['final_val_loss'] = best_val_loss if best_val_loss != float('inf') else float('nan')
    # Removed flops_g
    avg_final_metrics['inf_time_ms'] = inference_time_ms

    if vis_sample_collected:
        save_comparison_plot(vis_blurred, vis_output, vis_sharp, act_name, run_idx, avg_final_metrics, result_dir=result_dir)
        analyze_frequency_and_spatial(model, analyzer, vis_sharp, vis_output, vis_blurred, act_name, run_idx, result_dir=result_dir)

    print(f"--- EDSR {act_name} Run {run_idx+1} Final Avg Metrics ---")
    for key, value in avg_final_metrics.items(): print(f"  {key.upper()}: {value:.4f}")
    print("-------------------------------------")

    return avg_final_metrics

# --- Main function call ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*torch.load with weights_only=False.*")
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")
    warnings.filterwarnings("ignore", message="torch.meshgrid.*")

    generate_synthetic_dataset(force_generate=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Results will be saved in: {RESULT_DIR}") # EDSR specific

    try:
        full_dataset = PreGeneratedDeblurDataset(SYNTHETIC_DIR, num_samples=NUM_SAMPLES)
        if len(full_dataset) == 0: raise ValueError("Loaded dataset is empty.")
    except Exception as e: print(f"\nFATAL ERROR: Failed to load dataset from {SYNTHETIC_DIR}. Error: {e}"); exit(1)

    all_run_results = {run: {} for run in range(NUM_RUNS)}
    # Update metrics list (remove flops_g)
    metrics_to_aggregate = ["psnr", "ssim", "epi", "hf_recon", "lpips", "final_train_loss", "final_val_loss", "inf_time_ms"]

    for run_idx in range(NUM_RUNS):
        print(f"\n========== STARTING EDSR RUN {run_idx+1}/{NUM_RUNS} ==========")
        seed = run_idx
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        if len(full_dataset) < 5: print("FATAL ERROR: Dataset too small for split."); exit(1)
        test_size = max(1, int(TEST_SPLIT * len(full_dataset)))
        train_size = len(full_dataset) - test_size
        if train_size <=0: print("FATAL ERROR: No training samples after split."); exit(1)

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1); hpo_workers = 1
        pin_memory_flag = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=True if num_workers>0 and os.name!='nt' else False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag, persistent_workers=True if num_workers>0 and os.name!='nt' else False)

        hpo_train_loader = None; hpo_val_loader = None
        hpo_val_size = max(1, int(0.1 * len(train_dataset)))
        if len(train_dataset) > hpo_val_size:
             hpo_train_subset, hpo_val_subset = random_split(train_dataset, [len(train_dataset) - hpo_val_size, hpo_val_size], generator=torch.Generator().manual_seed(seed + 1))
             hpo_train_loader = DataLoader(hpo_train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=hpo_workers)
             hpo_val_loader = DataLoader(hpo_val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=hpo_workers)
        else: print("Warn: Using full train/test for HPO."); hpo_train_loader = train_loader; hpo_val_loader = test_loader

        for act_name, act_config in ACTIVATIONS.items():
            best_lr = DEFAULT_LR
            if HPO_ENABLED and hpo_train_loader is not None and hpo_val_loader is not None:
                print(f"\n--- Optimizing LR for EDSR/{act_name} (Run {run_idx+1}) ---")
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
                objective_func = lambda trial: objective(trial, act_name, act_config, hpo_train_loader, hpo_val_loader, device)
                try:
                    study.optimize(objective_func, n_trials=HPO_N_TRIALS, timeout=300)
                    if study.best_trial: best_lr = study.best_params["lr"]; print(f"--- Opt complete. Best LR: {best_lr:.2e} (Val: {study.best_value:.6f}) ---")
                    else: print(f"--- Opt yielded no valid trials for {act_name}. Using default LR. ---")
                except optuna.exceptions.TrialPruned: print(f"--- Optuna trial pruned for {act_name}. Trying default LR. ---")
                except Exception as e: print(f"Optuna optimization failed for {act_name}: {e}. Using default LR.")
                optuna.logging.set_verbosity(optuna.logging.INFO)
            elif not HPO_ENABLED: print(f"--- Skipping HPO for EDSR/{act_name}. Using default LR: {DEFAULT_LR:.2e} ---")
            else: print(f"--- Skipping HPO for EDSR/{act_name} due to invalid HPO loaders. Using default LR: {DEFAULT_LR:.2e} ---")

            current_wd = GLOBAL_WEIGHT_DECAY
            try:
                results = train_and_evaluate(act_name, act_config, best_lr, current_wd,
                                             train_loader, test_loader, run_idx, device, result_dir=RESULT_DIR)
                all_run_results[run_idx][act_name] = results
            except Exception as e:
                 print(f"\n!!! ERROR during training/evaluation for {act_name} Run {run_idx+1}: {e}")
                 traceback.print_exc()
                 all_run_results[run_idx][act_name] = {metric: float('nan') for metric in metrics_to_aggregate}

        print(f"========== COMPLETED EDSR RUN {run_idx+1}/{NUM_RUNS} ==========")

    # --- Aggregate results ---
    aggregated_results = {}
    for act_name in ACTIVATIONS.keys():
        aggregated_results[act_name] = {}
        for metric in metrics_to_aggregate:
            values = [all_run_results[run].get(act_name, {}).get(metric, float('nan')) for run in range(NUM_RUNS)]
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                aggregated_results[act_name][f"{metric}_mean"] = np.mean(valid_values)
                aggregated_results[act_name][f"{metric}_std"] = np.std(valid_values)
            else:
                 aggregated_results[act_name][f"{metric}_mean"] = float('nan')
                 aggregated_results[act_name][f"{metric}_std"] = float('nan')

    # --- Print and save aggregated results ---
    print("\n=== Aggregated EDSR Activation Function Results (Mean +/- Std Dev over Runs) ===")
    sort_key_metric = 'final_val_loss_mean'; reverse_sort = False
    valid_act_names_final = [name for name, res in aggregated_results.items() if not np.isnan(res.get(sort_key_metric, float('nan')))]
    ranked_activations = sorted(valid_act_names_final, key=lambda x: aggregated_results[x].get(sort_key_metric, float('inf') if not reverse_sort else float('-inf')), reverse=reverse_sort)

    # Update CSV Header
    csv_header = "Activation," + ",".join([f"{m}_mean,{m}_std" for m in metrics_to_aggregate]) + "\n"
    csv_lines = [csv_header]
    print("\n--- Aggregated Table Data ---")
    for act_name in ACTIVATIONS.keys(): # Iterate original order
        res = aggregated_results.get(act_name, {})
        print(f"\n--- {act_name} ---")
        line = f"{act_name},"
        for metric in metrics_to_aggregate:
            mean_val = res.get(f"{metric}_mean", float('nan')); std_val = res.get(f"{metric}_std", float('nan'))
            if not np.isnan(mean_val): print(f"  {metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}")
            else: print(f"  {metric.upper()}: nan")
            line += f"{mean_val:.6f},{std_val:.6f},"
        csv_lines.append(line.strip(',') + "\n")

    # Save to EDSR specific results file
    results_csv_path = os.path.join(RESULT_DIR, "metrics", "edsr_aggregated_results.csv")
    try:
        with open(results_csv_path, "w") as f: f.writelines(csv_lines)
        print(f"\nAggregated EDSR results saved to: {results_csv_path}")
    except IOError as e: print(f"\nError saving EDSR results to CSV: {e}")

    # --- Print rankings ---
    print("\n--- Final Rankings (Based on Mean Validation Loss) ---")
    for rank, act_name in enumerate(ranked_activations, 1):
         mean_val = aggregated_results[act_name].get(sort_key_metric, float('nan'))
         print(f"{rank}. {act_name}: {mean_val:.6f}")

    print("\n=== Rankings by other metrics (EDSR, using mean values) ===")
    metrics_for_ranking = {'epi_mean': True, 'hf_recon_mean': True, 'lpips_mean': False}
    for metric, higher_is_better in metrics_for_ranking.items():
         default_val = float('-inf') if higher_is_better else float('inf')
         valid_activations_metric = [act for act in ACTIVATIONS.keys() if act in aggregated_results and not np.isnan(aggregated_results[act].get(metric, float('nan')))]
         ranked = sorted(valid_activations_metric, key=lambda x: aggregated_results[x].get(metric, default_val), reverse=higher_is_better)
         print(f"\nRanked by {metric.upper()}:")
         for i, name in enumerate(ranked, 1): print(f"{i}. {name}: {aggregated_results[name].get(metric, float('nan')):.4f}")

    #print("\n=== EDSR based AFs analysis completed ===")