import os, math, argparse, time, csv, warnings
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
_Y_COEF = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
@torch.no_grad()
def rgb_to_y(img: torch.Tensor):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    coef = _Y_COEF.to(img.device, img.dtype)
    y = (img * coef).sum(dim=1, keepdim=True)
    return y.squeeze(0) if img.shape[0] == 1 else y

def psnr_y(a: torch.Tensor, b: torch.Tensor):
    mse = torch.mean((rgb_to_y(a) - rgb_to_y(b))**2).item()
    return float("inf") if mse < 1e-12 else 10. * math.log10(1. / mse)

def ssim_y(a: torch.Tensor, b: torch.Tensor):
    from skimage.metrics import structural_similarity as ssim
    ay = rgb_to_y(a).clamp(0,1).cpu().numpy().squeeze()
    by = rgb_to_y(b).clamp(0,1).cpu().numpy().squeeze()
    v = ssim(ay, by, data_range=1., channel_axis=None)
    return 0. if v is None or np.isnan(v) else float(v)

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, np.float32) / 255.0
    ten = torch.from_numpy(arr.transpose(2,0,1)).to(device)
    return ten.unsqueeze(0)

def _grid(L, tile, step):
    coords = list(range(0, max(1, L - tile + 1), step))
    if coords[-1] + tile < L:
        coords.append(L - tile)
    return coords

@torch.no_grad()
def forward_tiled(model, img, tile=336, overlap=168, bf16=False):
    _, _, H, W = img.shape
    step = tile - overlap
    rows, cols = _grid(H, tile, step), _grid(W, tile, step)
    out  = torch.zeros_like(img)
    wmap = torch.zeros((1,1,H,W), device=img.device)

    for top in rows:
        for left in cols:
            patch = img[..., top:top+tile, left:left+tile]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=bf16):
                pout = model(patch)
            out[...,  top:top+tile, left:left+tile] += pout
            wmap[..., top:top+tile, left:left+tile] += 1
    return out / wmap.clamp_min(1e-6)

def adapt_prefix(state_dict, want_dp):
    has_dp = next(iter(state_dict)).startswith("module.")
    if has_dp and not want_dp:
        return {k[7:]: v for k, v in state_dict.items()}
    if (not has_dp) and want_dp:
        return {"module."+k: v for k, v in state_dict.items()}
    return state_dict

def load_model(weights_path, device):
    try:
        from models import BVIFormer as Net
    except Exception:
        from models import DehazeFormer as Net
    model = Net().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
    sd = adapt_prefix(sd, want_dp=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

def list_images(d):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    return sorted([p for p in glob(os.path.join(d, "*")) if p.lower().endswith(exts)])

def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def build_pairs(hazy_dir, gt_dir):
    hazy_list = list_images(hazy_dir)
    gt_list   = list_images(gt_dir)
    gt_map    = {stem(p): p for p in gt_list}
    pairs, miss = [], []
    for hp in hazy_list:
        s = stem(hp)
        target = gt_map.get(s)
        if target is None:
            s0 = s.split("_")[0]
            target = gt_map.get(s0)
        if target is None:
            miss.append(hp)
            continue
        pairs.append((hp, target))
    return pairs, miss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazy_dir",  required=True)
    ap.add_argument("--gt_dir",    required=True)
    ap.add_argument("--weights",   required=True)
    ap.add_argument("--output_csv", default="scores.csv")
    ap.add_argument("--tile", type=int, default=336)
    ap.add_argument("--overlap", type=int, default=168)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    model  = load_model(args.weights, device)
    pairs, miss = build_pairs(args.hazy_dir, args.gt_dir)
    if len(pairs) == 0:
        raise RuntimeError("no paired hazy/gt image, please check it over.")
    if miss:
        print(f"[WARN] {len(miss)} hazy is not paired, will be skipped. Exsample:{os.path.basename(miss[0])}")
    t0 = time.time()
    rows = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    for i, (hazy_p, gt_p) in enumerate(pairs, 1):
        x = load_image(hazy_p, device).clamp(0,1)
        g = load_image(gt_p,   device).clamp(0,1)
        with torch.no_grad():
            y = forward_tiled(model, x, tile=args.tile, overlap=args.overlap, bf16=args.bf16).clamp(0,1)
        p = psnr_y(y, g)
        s = ssim_y(y, g)
        psnr_sum += p
        ssim_sum += s
        rows.append([os.path.basename(hazy_p), f"{p:.4f}", f"{s:.4f}"])
        if i % 10 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {os.path.basename(hazy_p)}  PSNR_Y={p:.3f}  SSIM_Y={s:.4f}")
    psnr_avg = psnr_sum / len(pairs)
    ssim_avg = ssim_sum / len(pairs)
    need_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "psnr_y", "ssim_y"])
        w.writerows(rows)
        w.writerow(["AVERAGE", f"{psnr_avg:.4f}", f"{ssim_avg:.4f}"])
    print(f"==> AVERAGE: PSNR_Y={psnr_avg:.3f} dB, SSIM_Y={ssim_avg:.4f} | {time.time()-t0:.1f}s")
    print(f"Scores saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
