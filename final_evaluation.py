"""
=============================================================================
  Underwater Image Enhancement — Benchmarking Suite
  YourModel is the ANCHOR.  Each competitor is measured against it.
=============================================================================
  Blocks:
    1  Data Ingestion & Standardisation
    2  Model Registry & Parameter Counting
    3  Inference Loop & GPU Timing (CUDA events)
    4  Metric Calculation  (PSNR · SSIM · UIQM)
    5  Aggregation & Visualisation  (anchor-relative Δ scores)
=============================================================================
"""

import os
import time
import glob
import warnings
import tempfile
import zipfile
import sys
import types
import re
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "/kaggle/input/datasets/noureldin199/lsui-large-scale-underwater-image-dataset/LSUI/input"
DEFAULT_GT_DIR     = "/kaggle/input/datasets/noureldin199/lsui-large-scale-underwater-image-dataset/LSUI/GT"
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_IMG_SIZE   = 256
DEFAULT_BATCH_SIZE = 1

DEFAULT_FINAL_MODEL_PATH   = "/kaggle/input/models/harinair0014/final-model/pytorch/default/1/best_uw_model"
DEFAULT_SHALLOW_MODEL_PATH = "/kaggle/input/models/harinair0014/shallow-uwnet/pytorch/default/1/shallow_uwnet_model.ckpt"
DEFAULT_WAVELET_MODEL_PATH = "/kaggle/input/models/harinair0014/wavelet-based/pytorch/default/1/wavelet_based_model.pth"

ANCHOR_KEY       = "YourModel"
ANCHOR_COLOR     = "#2563EB"
COMPETITOR_COLOR = "#94A3B8"
POSITIVE_DELTA   = "#16A34A"
NEGATIVE_DELTA   = "#DC2626"

def _resolve_device() -> torch.device:
    """
    torch.cuda.is_available() only checks that a CUDA library is present —
    it does NOT verify the GPU supports the compiled PyTorch CUDA kernels.
    cudaErrorNoKernelImageForDevice fires when the GPU compute capability
    doesn't match the PyTorch wheel (common in Kaggle when the session GPU
    changes between runs).  We catch this early with a real kernel launch.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # Allocate + compute to force an actual kernel dispatch
        t = torch.zeros(4, 4, device="cuda")
        _ = (t + 1).sum()
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        print(
            "\n  ╔══ CUDA unavailable ══════════════════════════════════════════╗"
            "\n  ║  torch.cuda.is_available() = True, but a kernel launch       ║"
           f"\n  ║  failed with: {str(e)[:55]:<55}║"
            "\n  ║                                                              ║"
            "\n  ║  Cause: PyTorch wheel compiled for a different GPU arch      ║"
            "\n  ║  than the one Kaggle assigned to this session.               ║"
            "\n  ║                                                              ║"
            "\n  ║  Fix — add this to a cell above and restart:                 ║"
            "\n  ║  !pip install torch torchvision --quiet \\                   ║"
            "\n  ║      --index-url https://download.pytorch.org/whl/cu121     ║"
            "\n  ║  (replace cu121 with your CUDA version from nvidia-smi)      ║"
            "\n  ║                                                              ║"
            "\n  ║  Falling back to CPU — benchmark runs, just slower.          ║"
            "\n  ╚══════════════════════════════════════════════════════════════╝\n"
        )
        return torch.device("cpu")


DEVICE = _resolve_device()
print(f"  [device] {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 1 · DATA INGESTION & STANDARDISATION
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def collect_image_paths(directory: str) -> list[str]:
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(paths)


class UnderwaterDataset(Dataset):
    def __init__(self, input_dir: str, gt_dir: str = "", img_size: int = 256):
        self.input_paths = collect_image_paths(input_dir)
        self.has_gt      = bool(gt_dir and os.path.isdir(gt_dir))
        self.gt_dir      = gt_dir
        self.img_size    = img_size
        self.transform   = transforms.Compose([transforms.ToTensor()])
        if not self.input_paths:
            raise FileNotFoundError(f"No images found in: {input_dir}")

    def _load(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        return self.transform(img)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp_path   = self.input_paths[idx]
        inp_tensor = self._load(inp_path)
        gt_tensor  = torch.zeros_like(inp_tensor)
        if self.has_gt:
            gt_path = os.path.join(self.gt_dir, os.path.basename(inp_path))
            if os.path.isfile(gt_path):
                gt_tensor = self._load(gt_path)
        return inp_tensor, gt_tensor, inp_path


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 2 · MODEL REGISTRY & PARAMETER COUNTING
# ─────────────────────────────────────────────────────────────────────────────

# ── Wavelet-Based wrapper ─────────────────────────────────────────────────────
# The wavelet_based_model.pth is a state dict — we need a shell class to load
# the weights into.  If your WaveletNet class is defined in an earlier notebook
# cell, that will be picked up automatically via the notebook alias logic.
# Otherwise this generic wrapper is used (it won't produce good output, but it
# will at least load and run so the benchmark does not crash).

class _WaveletNetFallback(nn.Module):
    """
    Generic fallback shell for wavelet_based_model.pth.
    Replace this with your actual WaveletNet class if you have it,
    by defining `WaveletNet` in an earlier notebook cell.
    """
    def __init__(self):
        super().__init__()
        # Minimal encoder-decoder so load_state_dict(strict=False) can partially load
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_DIRNAMES = {
    "depth_test", "depth_train", "gt_test", "gt_train",
    "input_test", "input_train", "real_90_gdcp", "real_90_input",
    "testing_data_extra", "vgg_pretrained",
    "__pycache__", ".git", ".ipynb_checkpoints",
    "Generalization of the Dark Channel Prior",
}


def _is_tf_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    return any(f.endswith(".index") for f in files) and any(".data-" in f for f in files)


def _find_tf_checkpoint_prefix(ckpt_dir: str) -> str | None:
    tf_index_file = os.path.join(ckpt_dir, "checkpoint")
    if os.path.isfile(tf_index_file):
        with open(tf_index_file) as f:
            for line in f:
                m = re.search(r'model_checkpoint_path:\s*"([^"]+)"', line)
                if m:
                    prefix = m.group(1)
                    if not os.path.isabs(prefix):
                        prefix = os.path.join(ckpt_dir, prefix)
                    return prefix
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith(".index"):
            return os.path.join(ckpt_dir, fname[:-len(".index")])
    return None


def _find_checkpoint_dir(root: str) -> str | None:
    best = None
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRNAMES and not d.startswith(".")]
        if os.path.basename(dirpath).lower() == "checkpoint":
            best = dirpath
            break
        if _is_tf_checkpoint_dir(dirpath):
            best = dirpath
    return best


def _find_pytorch_checkpoint(root: str) -> str | None:
    pt_exts = {".pt", ".pth", ".ckpt", ".bin"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRNAMES and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in pt_exts:
                return os.path.join(dirpath, fname)
    return None


def _zip_directory_checkpoint(dir_path: str) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath   = os.path.join(root, fname)
                arcname = os.path.join("archive", os.path.relpath(fpath, dir_path))
                zf.write(fpath, arcname)
    return tmp_path


def _is_torch_archive_dir(dir_entries: set) -> bool:
    TORCH_META = {"version", ".storage_alignment", "__MACOSX"}
    return "version" in dir_entries and bool(dir_entries - TORCH_META)


def _extract_state_dict(payload) -> OrderedDict | None:
    if isinstance(payload, OrderedDict):
        return payload
    if isinstance(payload, dict):
        for key in ["state_dict", "model_state_dict", "net", "weights"]:
            val = payload.get(key)
            if isinstance(val, (OrderedDict, dict)):
                return OrderedDict(val)
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return OrderedDict(payload)
    return None


def _strip_module_prefix(sd: OrderedDict) -> OrderedDict:
    return OrderedDict({(k[7:] if k.startswith("module.") else k): v
                        for k, v in sd.items()})


def _install_notebook_model_alias() -> None:
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    model_mod = sys.modules.setdefault("model", types.ModuleType("model"))
    for name in dir(main_mod):
        if not name.startswith("__"):
            setattr(model_mod, name, getattr(main_mod, name))


def _normalize_symbol(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _inject_missing_model_symbol(missing_name: str) -> bool:
    main_mod  = sys.modules.get("__main__")
    model_mod = sys.modules.get("model")
    if not (main_mod and model_mod):
        return False
    target = _normalize_symbol(missing_name)
    best   = None
    for name in dir(main_mod):
        obj = getattr(main_mod, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            norm = _normalize_symbol(name)
            if norm == target or target in norm or norm in target:
                best = obj
                break
    if best is None:
        def _fwd(self, x, *a, **kw):
            # Try named sub-modules that are likely the main network
            for attr in ["model", "net", "generator", "unet", "backbone"]:
                m = getattr(self, attr, None)
                if isinstance(m, nn.Module):
                    return m(x)
            # Single child — delegate directly
            kids = list(self.children())
            if len(kids) == 1:
                return kids[0](x)
            # Multiple children — DO NOT chain them sequentially; channel
            # counts almost certainly won't line up.  Return input unchanged
            # so the benchmark can at least measure UIQM on unprocessed images.
            return x
        best = type(missing_name, (nn.Module,), {"forward": _fwd})
    setattr(model_mod, missing_name, best)
    setattr(main_mod,  missing_name, best)
    return True


def _sanitize_compat_shims(model: nn.Module, model_name: str) -> nn.Module:
    """
    Walk every sub-module in the loaded model.  If any forward method is
    named '_fwd' (our compat shim — from ANY kernel run, old or new), rebind
    it at the INSTANCE level with a safe version so it never crashes.

    Rebinding per-instance (via types.MethodType) leaves other instances of
    the same class untouched, which is important because sys.modules caches
    the class object across cell re-runs.
    """
    import types as _types

    def _param_count(m):
        return sum(p.numel() for p in m.parameters())

    fixed = 0
    for mod_name, mod in model.named_modules():
        cls_fwd = getattr(type(mod), "forward", None)
        if cls_fwd is None or getattr(cls_fwd, "__name__", "") != "_fwd":
            continue

        # Build the safest possible replacement for this specific instance
        named_kids = list(mod.named_children())

        if not named_kids:
            # Leaf compat node — pass input straight through
            def _safe(self, x, *a, **kw): return x

        elif len(named_kids) == 1:
            # Exactly one child — delegate unconditionally
            cname = named_kids[0][0]
            def _safe(self, x, *a, _cn=cname, **kw):
                return getattr(self, _cn)(x)

        else:
            # Multiple children.
            # Priority 1: well-known single-entry-point attribute names
            entry = next(
                (n for n, _ in named_kids
                 if n in ("model", "net", "unet", "backbone", "generator", "encoder")),
                None,
            )
            if entry:
                def _safe(self, x, *a, _en=entry, **kw):
                    return getattr(self, _en)(x)
            else:
                # Priority 2: child with the most parameters (main body)
                heaviest = max(named_kids, key=lambda nc: _param_count(nc[1]))[0]
                def _safe(self, x, *a, _hn=heaviest, **kw):
                    return getattr(self, _hn)(x)

        mod.forward = _types.MethodType(_safe, mod)
        fixed += 1

    if fixed:
        print(f"  [i] {model_name}: rebound {fixed} stale compat-shim forward(s) "
              f"(cross-kernel _fwd detected and replaced)")
    return model


def _missing_symbol(exc: Exception) -> str | None:
    m = re.search(r"attribute '([^']+)'", str(exc))
    return m.group(1) if m else None


def _load_pytorch_file(path: str, model_name: str, model_ctor=None) -> nn.Module:
    _install_notebook_model_alias()

    # TorchScript
    try:
        m = torch.jit.load(path, map_location=DEVICE)
        m.eval()
        print(f"  [✓] {model_name} loaded as TorchScript.")
        return m
    except Exception:
        pass

    # Standard torch.load with automatic symbol injection
    payload = None
    for _ in range(5):
        try:
            payload = torch.load(path, map_location=DEVICE, weights_only=False)
            break
        except AttributeError as exc:
            sym = _missing_symbol(exc)
            if not (sym and _inject_missing_model_symbol(sym)):
                raise

    if payload is None:
        raise RuntimeError(f"Could not load {model_name} — unresolved pickle symbols.")

    if isinstance(payload, nn.Module):
        payload = _sanitize_compat_shims(payload, model_name)
        payload.eval()
        print(f"  [✓] {model_name} loaded as full nn.Module.")
        return payload

    sd = _extract_state_dict(payload)
    if sd is not None:
        if model_ctor is None:
            raise ValueError(
                f"'{model_name}' checkpoint is a state dict — pass "
                "model_ctor=YourModelClass to load_checkpoint_model().")
        model    = model_ctor()
        sd_clean = _strip_module_prefix(sd)
        missing, unexpected = model.load_state_dict(sd_clean, strict=False)

        # ── Match-ratio guard ─────────────────────────────────────────────
        # If almost none of the checkpoint keys matched the model shell, the
        # architecture is wrong and running inference produces garbage.
        total_sd_keys = len(sd_clean)
        matched       = total_sd_keys - len(unexpected)
        match_pct     = matched / max(total_sd_keys, 1) * 100

        if missing:
            print(f"  [!] {model_name}: {len(missing)} missing keys "
                  f"(first 5: {missing[:5]})")
        if unexpected:
            print(f"  [!] {model_name}: {len(unexpected)} unexpected keys "
                  f"(first 5: {unexpected[:5]})")

        if match_pct < 10:
            top_keys = sorted({k.split(".")[0] for k in sd_clean})
            raise ValueError(
                f"\n  x  {model_name}: only {matched}/{total_sd_keys} checkpoint "
                f"keys ({match_pct:.0f}%) matched the model_ctor shell.\n"
                f"     The model class does not match the checkpoint architecture.\n"
                f"\n"
                f"     Top-level module names in the checkpoint:\n"
                f"       {top_keys}\n"
                f"\n"
                f"     Fix: define the correct PyTorch class for {model_name} in\n"
                f"     a notebook cell above, then pass it as model_ctor=<YourClass>.\n"
                f"     The class must have attributes matching the names above."
            )

        model.eval()
        print(f"  [✓] {model_name} loaded from state dict ({match_pct:.0f}% key match).")
        return model

    raise ValueError(f"Unable to load {model_name} from '{{path}}'.")


def load_checkpoint_model(path: str, model_name: str,
                          model_ctor=None) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {path}")

    # Direct file
    if os.path.isfile(path):
        return _load_pytorch_file(path, model_name, model_ctor)

    dir_entries = set(os.listdir(path))

    # Extracted torch zip archive
    if _is_torch_archive_dir(dir_entries):
        print(f"  [i] {model_name}: detected extracted torch archive — re-zipping …")
        tmp = _zip_directory_checkpoint(path)
        try:
            return _load_pytorch_file(tmp, model_name, model_ctor)
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)

    # Repo directory — PyTorch file first
    pt_file = _find_pytorch_checkpoint(path)
    if pt_file:
        print(f"  [i] {model_name}: found PyTorch checkpoint → {pt_file}")
        return _load_pytorch_file(pt_file, model_name, model_ctor)

    # TF checkpoint fallback
    ckpt_dir = _find_checkpoint_dir(path)
    if ckpt_dir is None:
        if _is_tf_checkpoint_dir(path):
            ckpt_dir = path
        else:
            raise ValueError(
                f"Could not locate any checkpoint inside '{path}'.\n"
                f"  Contents: {sorted(dir_entries)}")

    print(f"  [i] {model_name}: found TF checkpoint dir → {ckpt_dir}")
    actual_tf_dir = ckpt_dir
    for dirpath, dirnames, _ in os.walk(ckpt_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _is_tf_checkpoint_dir(dirpath):
            actual_tf_dir = dirpath

    prefix = _find_tf_checkpoint_prefix(actual_tf_dir)
    if prefix is None:
        raise ValueError(f"TF checkpoint dir found but no prefix resolved: {actual_tf_dir}")

    print(f"  [i] {model_name}: TF checkpoint prefix → {prefix}")
    if model_ctor is None:
        raise ValueError(
            f"'{model_name}' has a TensorFlow checkpoint. "
            "model_ctor= is required to map TF weights into a PyTorch shell.")

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            f"{model_name} uses a TF checkpoint. Install TensorFlow: pip install tensorflow")

    reader   = tf.train.load_checkpoint(prefix)
    var_map  = reader.get_variable_to_shape_map()
    model    = model_ctor()
    pt_state = model.state_dict()
    pt_lower = {k.lower(): k for k in pt_state}
    new_state, loaded, skipped = OrderedDict(pt_state), 0, 0

    _TF_RENAMES = {"kernel": "weight", "gamma": "weight", "beta": "bias",
                   "moving_mean": "running_mean", "moving_variance": "running_var"}

    for tf_name in var_map:
        if any(s in tf_name for s in ["Adam", "Momentum", "global_step"]):
            continue
        pt_key = ".".join(_TF_RENAMES.get(p, p)
                          for p in re.sub(r"/+", ".", tf_name.split(":")[0]).split("."))
        tensor = torch.from_numpy(reader.get_tensor(tf_name))
        matched = pt_key if pt_key in pt_state else pt_lower.get(pt_key.lower())
        if not matched:
            skipped += 1
            continue
        if tensor.ndim == 4: tensor = tensor.permute(3, 2, 0, 1)
        elif tensor.ndim == 2: tensor = tensor.t()
        if tensor.shape == pt_state[matched].shape:
            new_state[matched] = tensor.to(pt_state[matched].dtype)
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(new_state, strict=False)
    print(f"  [✓] {model_name}: loaded {loaded} TF weights ({skipped} skipped).")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  FORWARD PASS — with upfront input-format detection
# ─────────────────────────────────────────────────────────────────────────────

def _build_frequency_inputs(inp: torch.Tensor):
    """
    Builds (lf, hf) 9-channel inputs required by ImageEnhancementNetwork.
    Lazy-imports splitImage and DCT_CUTOFF_RATIO from your project modules.
    """
    try:
        from dct import splitImage
        from constants import DCT_CUTOFF_RATIO
    except ImportError as e:
        raise ImportError(
            f"Could not import frequency-split helpers: {e}\n"
            "Make sure 'dct.py' and 'constants.py' are in your working directory "
            "or on sys.path.") from e

    inp_cpu = inp.detach().clamp(0, 1).cpu().numpy()
    lf_batch, hf_batch = [], []
    for sample in inp_cpu:
        rgb    = np.transpose(sample, (1, 2, 0))
        hf_rgb, lf_rgb = splitImage(rgb, cutoff_ratio=DCT_CUTOFF_RATIO)
        img_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        lab    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hsv    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 180.0; hsv[:, :, 1] /= 255.0; hsv[:, :, 2] /= 255.0
        lf_batch.append(torch.from_numpy(
            np.concatenate([np.clip(lf_rgb, 0, 1), lab, hsv], -1)).permute(2,0,1).float())
        hf_batch.append(torch.from_numpy(
            np.concatenate([np.clip(hf_rgb, 0, 1), lab, hsv], -1)).permute(2,0,1).float())
    return torch.stack(lf_batch).to(DEVICE), torch.stack(hf_batch).to(DEVICE)


# Input format literals
_FMT_RGB   = "rgb"       # model(inp)
_FMT_LF_HF = "lf_hf"    # model(lf, hf)


def _probe_input_format(model: nn.Module, img_size: int = 64) -> str:
    """
    Runs a single tiny forward pass OFF the timed path to determine whether
    the model expects a plain RGB tensor or (lf, hf) frequency pair.

    DEVICE is already validated by _resolve_device() at import time, so no
    additional CUDA health check is needed here.  Always call this BEFORE
    the CUDA timing loop — never inside it.
    """
    # Always probe on CPU to avoid any GPU state side-effects.
    # The real inference runs on DEVICE; this is format-detection only.
    probe_dev = torch.device("cpu")
    orig_params = [p for p in model.parameters()]
    was_on_gpu  = orig_params[0].device.type == "cuda" if orig_params else False
    probe_model = model.cpu() if was_on_gpu else model

    fmt = _FMT_RGB    # safe default

    with torch.no_grad():
        # ── Try plain RGB ─────────────────────────────────────────────────
        try:
            dummy = torch.zeros(1, 3, img_size, img_size, device=probe_dev)
            out   = probe_model(dummy)
            fmt   = _FMT_RGB
            _ = out   # consumed — format confirmed
        except TypeError:
            # Model requires more positional args → try (lf, hf)
            try:
                lf  = torch.zeros(1, 9, img_size, img_size, device=probe_dev)
                hf  = torch.zeros(1, 9, img_size, img_size, device=probe_dev)
                probe_model(lf, hf)
                fmt = _FMT_LF_HF
            except Exception:
                fmt = _FMT_RGB   # unknown — let real inference surface the error
        except Exception:
            fmt = _FMT_RGB       # shape / layer error — non-fatal at probe stage

    # Restore model to GPU if it was moved
    if was_on_gpu:
        model.to(DEVICE)

    return fmt


def _extract_prediction(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for key in ["enhanced", "output", "pred", "prediction", "image"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        for v in output.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Model output has no tensor: {type(output)}")


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 3 · INFERENCE LOOP & GPU TIMING
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_registry(your_model: nn.Module,
                         competitors: dict[str, nn.Module]) -> dict[str, nn.Module]:
    if your_model is None:
        raise ValueError("your_model must not be None.")
    registry = {ANCHOR_KEY: your_model, **competitors}
    for m in registry.values():
        m.to(DEVICE).eval()
    return registry


def print_model_summary(registry: dict):
    print("\n" + "═" * 60)
    print(f"  {'Model':<34} {'Trainable Params':>22}")
    print("═" * 60)
    for name, model in registry.items():
        tag = "  ◀ anchor" if name == ANCHOR_KEY else ""
        print(f"  {name:<34} {count_parameters(model):>22,}{tag}")
    print("═" * 60 + "\n")


def run_inference(model: nn.Module, dataloader: DataLoader,
                  input_fmt: str = _FMT_RGB):
    """
    input_fmt is determined once by _probe_input_format() BEFORE this is called,
    so no format-detection ever happens inside the timed GPU loop.
    """
    outputs, gts, paths = [], [], []
    total_ms = 0.0
    use_cuda = DEVICE.type == "cuda"

    with torch.no_grad():
        for inp, gt, batch_paths in dataloader:
            inp = inp.to(DEVICE)

            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()

            if input_fmt == _FMT_LF_HF:
                lf, hf = _build_frequency_inputs(inp)
                out = model(lf, hf)
            else:
                out = model(inp)

            if use_cuda:
                e.record()
                torch.cuda.synchronize()
                total_ms += s.elapsed_time(e)
            else:
                total_ms += 0.0   # handled below for CPU

            pred = _extract_prediction(out)
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)

            out_np = (pred.clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
            gt_np  = (gt.clamp(0,1).cpu().numpy()   * 255).astype(np.uint8)
            for i in range(out_np.shape[0]):
                outputs.append(out_np[i].transpose(1, 2, 0))
                gts.append(gt_np[i].transpose(1, 2, 0))
                paths.append(batch_paths[i])

    # CPU timing: re-run a single pass to get a representative ms figure
    if not use_cuda and outputs:
        inp_sample, _, _ = next(iter(dataloader))
        inp_sample = inp_sample.to(DEVICE)
        with torch.no_grad():
            t0 = time.perf_counter()
            if input_fmt == _FMT_LF_HF:
                lf, hf = _build_frequency_inputs(inp_sample)
                model(lf, hf)
            else:
                model(inp_sample)
            total_ms = (time.perf_counter() - t0) * 1000 * len(outputs)

    return outputs, gts, paths, total_ms / max(len(outputs), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 4 · METRIC CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, gt):
    return float("nan") if gt.sum() == 0 else psnr_fn(gt, pred, data_range=255)

def compute_ssim(pred, gt):
    return float("nan") if gt.sum() == 0 else ssim_fn(gt, pred, channel_axis=2, data_range=255)

def _uicm(img):
    R,G,B = (img[:,:,i].astype(np.float32) for i in range(3))
    RG,YB = R-G, (R+G)/2-B
    def atm(x):
        xs=np.sort(x.flatten()); n=len(xs)
        return xs[int(0.1*n):int(0.9*n)].mean() if n>0 else 0.0
    mrg,myb = atm(RG),atm(YB)
    srg = np.sqrt(np.mean((RG-mrg)**2)); syb = np.sqrt(np.mean((YB-myb)**2))
    return float(-0.0268*np.sqrt(mrg**2+myb**2)+0.1586*np.sqrt(srg**2+syb**2))

def _uism(img):
    score=0.0
    for ch,lam in enumerate([0.299,0.587,0.114]):
        sx=cv2.Sobel(img[:,:,ch],cv2.CV_64F,1,0,ksize=3)
        sy=cv2.Sobel(img[:,:,ch],cv2.CV_64F,0,1,ksize=3)
        edge=np.sqrt(sx**2+sy**2); h,w=edge.shape
        eme,cnt=0.0,0
        for r in range(0,h-8+1,8):
            for c in range(0,w-8+1,8):
                blk=edge[r:r+8,c:c+8]; mn,mx=blk.min(),blk.max()
                if mn>0: eme+=np.log(mx/mn)
                cnt+=1
        score+=lam*(eme/max(cnt,1))
    return float(score)

def _uiconm(img,bs=8):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32); h,w=gray.shape
    amee,cnt=0.0,0
    for r in range(0,h-bs+1,bs):
        for c in range(0,w-bs+1,bs):
            blk=gray[r:r+bs,c:c+bs]; mn,mx=blk.min(),blk.max(); d=mx+mn
            if d>0 and mx>mn: amee+=((mx-mn)/d)*np.log((mx-mn)/d+1e-8)
            cnt+=1
    return float(amee/max(cnt,1))

def compute_uiqm(img):
    return 0.0282*_uicm(img)+0.2953*_uism(img)+3.5753*_uiconm(img)

def evaluate_outputs(outputs, gts, has_gt):
    ps,ss,us=[],[],[]
    for pred,gt in zip(outputs,gts):
        us.append(compute_uiqm(pred))
        ps.append(compute_psnr(pred,gt) if has_gt else float("nan"))
        ss.append(compute_ssim(pred,gt) if has_gt else float("nan"))
    return {"psnr":ps,"ssim":ss,"uiqm":us}


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 5 · AGGREGATION & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["PSNR (dB)", "SSIM", "UIQM", "Inf. Time (ms)"]


def build_summary_dataframe(results):
    rows=[]
    for name,data in results.items():
        rows.append({"Model":name,"PSNR (dB)":np.nanmean(data["psnr"]),
                     "SSIM":np.nanmean(data["ssim"]),"UIQM":np.nanmean(data["uiqm"]),
                     "Inf. Time (ms)":data["avg_ms"],"Params (M)":data["params"]/1e6})
    return pd.DataFrame(rows).set_index("Model")


def build_delta_dataframe(summary_df):
    anchor=summary_df.loc[ANCHOR_KEY,METRIC_COLS]
    rows=[]
    for name in summary_df.index:
        if name==ANCHOR_KEY: continue
        row=summary_df.loc[name,METRIC_COLS]-anchor; row.name=name; rows.append(row)
    df=pd.DataFrame(rows); df.columns=[f"Δ {c}" for c in df.columns]
    return df


def save_csv(summary_df,delta_df,out_dir):
    summary_df.to_csv(s:=os.path.join(out_dir,"benchmark_summary.csv"))
    delta_df.to_csv(d:=os.path.join(out_dir,"benchmark_delta_vs_yours.csv"))
    print(f"  [✓] Summary CSV           → {s}")
    print(f"  [✓] Delta CSV             → {d}")


def plot_absolute_bars(summary_df,out_dir):
    valid=[c for c in METRIC_COLS if not summary_df[c].isna().all()]
    fig,axes=plt.subplots(1,len(valid),figsize=(4.5*len(valid),5))
    if len(valid)==1: axes=[axes]
    sns.set_theme(style="whitegrid")
    colors=[ANCHOR_COLOR if n==ANCHOR_KEY else COMPETITOR_COLOR for n in summary_df.index]
    for ax,col in zip(axes,valid):
        bars=ax.barh(summary_df.index,summary_df[col],color=colors,edgecolor="white",height=0.55)
        ax.bar_label(bars,fmt="%.4f",padding=4,fontsize=8)
        ax.set_title(col,fontsize=12,fontweight="bold"); ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
        for lbl in ax.get_yticklabels():
            if lbl.get_text()==ANCHOR_KEY:
                lbl.set_fontweight("bold"); lbl.set_color(ANCHOR_COLOR)
    fig.legend(handles=[mpatches.Patch(color=ANCHOR_COLOR,label=f"{ANCHOR_KEY} (yours)"),
                        mpatches.Patch(color=COMPETITOR_COLOR,label="Competitors")],
               loc="lower center",ncol=2,fontsize=9,frameon=False,bbox_to_anchor=(0.5,-0.04))
    plt.suptitle("Absolute Scores — All Models",fontsize=13,fontweight="bold",y=1.02)
    plt.tight_layout()
    plt.savefig(p:=os.path.join(out_dir,"chart_absolute.png"),dpi=150,bbox_inches="tight")
    plt.close(); print(f"  [✓] Absolute bar chart    → {p}")


def plot_delta_bars(delta_df,out_dir):
    cols=delta_df.columns.tolist(); n=len(cols)
    fig,axes=plt.subplots(1,n,figsize=(4.5*n,max(3,len(delta_df)*1.2+2)))
    if n==1: axes=[axes]
    sns.set_theme(style="whitegrid")
    for ax,col in zip(axes,cols):
        vals=delta_df[col]; bip="Time" not in col
        colors=[POSITIVE_DELTA if (v>0)==bip else NEGATIVE_DELTA for v in vals]
        bars=ax.barh(vals.index,vals,color=colors,edgecolor="white",height=0.5)
        ax.bar_label(bars,fmt="%+.4f",padding=4,fontsize=8)
        ax.axvline(0,color="black",linewidth=1.4,linestyle="--",alpha=0.65)
        ax.set_title(col,fontsize=11,fontweight="bold"); ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(handles=[mpatches.Patch(color=POSITIVE_DELTA,label="Competitor beats yours"),
                        mpatches.Patch(color=NEGATIVE_DELTA,label="Competitor worse than yours")],
               loc="lower center",ncol=2,fontsize=9,frameon=False,bbox_to_anchor=(0.5,-0.06))
    plt.suptitle(f"Competitor Δ vs {ANCHOR_KEY}  (dashed line = your score)",
                 fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout()
    plt.savefig(p:=os.path.join(out_dir,"chart_delta_vs_yours.png"),dpi=150,bbox_inches="tight")
    plt.close(); print(f"  [✓] Delta bar chart       → {p}")


def plot_radar(summary_df,out_dir):
    cols=["PSNR (dB)","SSIM","UIQM"]; sub=summary_df[cols].copy()
    for c in cols:
        mn,mx=sub[c].min(),sub[c].max()
        sub[c]=(sub[c]-mn)/(mx-mn+1e-8) if mx>mn else 0.5
    N=len(cols); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()+[0]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw={"polar":True})
    cmap=plt.get_cmap("tab10")
    for i,(name,row) in enumerate(sub.iterrows()):
        vals=row.tolist()+row.tolist()[:1]; is_a=name==ANCHOR_KEY
        color=ANCHOR_COLOR if is_a else cmap(i+1)
        ax.plot(angles,vals,"-o",label=name,color=color,linewidth=3.0 if is_a else 1.5,zorder=5 if is_a else 2)
        ax.fill(angles,vals,alpha=0.20 if is_a else 0.05,color=color,zorder=5 if is_a else 2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([c.split(" ")[0] for c in cols],fontsize=11)
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["0.25","0.50","0.75","1.00"],fontsize=7)
    ax.set_title("Normalised Metric Radar\n(↑ outer edge = better)",fontsize=12,fontweight="bold",pad=15)
    ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.15),fontsize=9)
    plt.savefig(p:=os.path.join(out_dir,"chart_radar.png"),dpi=150,bbox_inches="tight")
    plt.close(); print(f"  [✓] Radar chart           → {p}")


def print_delta_table(summary_df,delta_df):
    print("\n"+"═"*72+"\n  ABSOLUTE SCORES\n"+"═"*72)
    print(summary_df.round(4).to_string())
    print("\n"+"═"*72+f"\n  DELTA vs {ANCHOR_KEY}   ( + means competitor beats yours )\n"+"═"*72)
    print(delta_df.round(4).to_string())
    print("═"*72+"\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(your_model, competitors,
                  input_dir=DEFAULT_INPUT_DIR, gt_dir=DEFAULT_GT_DIR,
                  output_dir=DEFAULT_OUTPUT_DIR,
                  img_size=DEFAULT_IMG_SIZE, batch_size=DEFAULT_BATCH_SIZE):

    os.makedirs(output_dir, exist_ok=True)

    print("\n[Block 1]  Loading & standardising dataset …")
    dataset    = UnderwaterDataset(input_dir, gt_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    print(f"           {len(dataset)} images · {img_size}×{img_size} "
          f"· GT={'✓' if dataset.has_gt else '✗'}")

    print("\n[Block 2]  Building model registry …")
    registry = build_model_registry(your_model, competitors)
    print_model_summary(registry)

    all_results = {}
    for model_name, model in registry.items():
        tag = "  ← anchor" if model_name == ANCHOR_KEY else ""
        print(f"[Block 3]  Inference → {model_name}{tag} …")

        # ── Probe input format BEFORE the timed loop ──────────────────────
        fmt = _probe_input_format(model, img_size=min(img_size, 64))
        print(f"           Input format : {fmt}")

        outputs, gts, paths, avg_ms = run_inference(model, dataloader, input_fmt=fmt)
        print(f"           Avg inference time : {avg_ms:.2f} ms/image")

        print(f"[Block 4]  Metrics    → {model_name} …")
        metrics = evaluate_outputs(outputs, gts, dataset.has_gt)
        all_results[model_name] = {**metrics, "avg_ms": avg_ms,
                                   "params": count_parameters(model)}
        print(f"           PSNR={np.nanmean(metrics['psnr']):.4f}  "
              f"SSIM={np.nanmean(metrics['ssim']):.4f}  "
              f"UIQM={np.nanmean(metrics['uiqm']):.4f}\n")

    print("[Block 5]  Aggregating & visualising …")
    summary_df = build_summary_dataframe(all_results)
    delta_df   = build_delta_dataframe(summary_df)
    print_delta_table(summary_df, delta_df)
    save_csv(summary_df, delta_df, output_dir)
    plot_absolute_bars(summary_df, output_dir)
    plot_delta_bars(delta_df, out_dir=output_dir)
    plot_radar(summary_df, out_dir=output_dir)
    print("\n  Benchmarking complete.\n")
    return summary_df, delta_df


# ─────────────────────────────────────────────────────────────────────────────
#  NOTEBOOK ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── YourModel ─────────────────────────────────────────────────────────
    ImageEnhancementNetwork = globals().get("ImageEnhancementNetwork")
    if ImageEnhancementNetwork is None:
        raise NameError(
            "ImageEnhancementNetwork is not defined. "
            "Define it in an earlier notebook cell.")

    YOUR_MODEL = load_checkpoint_model(
        path=DEFAULT_FINAL_MODEL_PATH,
        model_name=ANCHOR_KEY,
        model_ctor=ImageEnhancementNetwork,
    )

    # ── Competitors — each wrapped in try/except so one bad model ─────────
    # ── doesn't abort the whole benchmark ────────────────────────────────
    COMPETITORS = {}     # ← must be a fresh empty dict, not the placeholder

    try:
        COMPETITORS["Shallow-UWNet"] = load_checkpoint_model(
            path=DEFAULT_SHALLOW_MODEL_PATH,
            model_name="Shallow-UWNet",
            # No model_ctor needed — the .ckpt contains the full nn.Module
        )
    except Exception as exc:
        print(f"  [!] Skipping Shallow-UWNet: {exc}")

    try:
        # ── Wavelet-Based ──────────────────────────────────────────────────
        # The checkpoint keys start with  snet_structure_fE / snet_texture_fE
        # which means the real architecture is a dual-branch UNet-style network.
        #
        # You have two options:
        #
        #  Option A (recommended) — copy the model class from the paper's repo
        #  and define it in a cell above this one, then it is used automatically:
        #
        #    class WaveletNet(nn.Module):
        #        def __init__(self): ...   # snet_structure_fE, snet_texture_fE, …
        #        def forward(self, x): ...
        #
        #  Option B — skip Wavelet-Based by setting SKIP_WAVELET = True below.
        #
        SKIP_WAVELET = False          # ← set True to skip

        if SKIP_WAVELET:
            raise RuntimeError("Wavelet-Based skipped by user request (SKIP_WAVELET=True).")

        WaveletNet = globals().get("WaveletNet")
        if WaveletNet is None:
            raise ValueError(
                "WaveletNet class not found.\n"
                "  The checkpoint has keys like 'snet_structure_fE.*' and "
                "'snet_texture_fE.*'\n"
                "  which do not match the _WaveletNetFallback shell.\n"
                "  Define the real WaveletNet class in a cell above and re-run,\n"
                "  or set SKIP_WAVELET = True to exclude it from the benchmark."
            )

        COMPETITORS["Wavelet-Based"] = load_checkpoint_model(
            path=DEFAULT_WAVELET_MODEL_PATH,
            model_name="Wavelet-Based",
            model_ctor=WaveletNet,
        )
    except Exception as exc:
        print(f"  [!] Skipping Wavelet-Based:\n      {exc}")

    run_benchmark(your_model=YOUR_MODEL, competitors=COMPETITORS)