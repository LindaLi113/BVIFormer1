import os
import re
import csv
import glob
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from models.bviformer import BVIFormer
def tensor_to_uint8_rgb(t: torch.Tensor):
    """[3,H,W] -> uint8 HxWx3"""
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)

def compute_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)

def compute_ssim(pred, gt):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    gt_np   = gt.detach().cpu().numpy().transpose(1, 2, 0)
    val = ssim(pred_np, gt_np, data_range=1.0, channel_axis=-1)
    if val is None or np.isnan(val):
        return 0.0
    return float(val)

def save_tensor_as_image(t_3hw: torch.Tensor, path: str):
    im = Image.fromarray(tensor_to_uint8_rgb(t_3hw))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)

def draw_text(im: Image.Image, text: str) -> Image.Image:
    im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, im.size[1] // 40))
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        w, h = (r - l), (b - t)
    except Exception:
        w, h = draw.textsize(text, font=font)
    try:
        overlay = im.convert('RGBA')
        alpha = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
        bg = Image.new('RGBA', (w + pad * 2, h + pad * 2), (0, 0, 0, 160))
        alpha.paste(bg, (0, 0))
        overlay = Image.alpha_composite(overlay, alpha)
        draw = ImageDraw.Draw(overlay)
        draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
        return overlay.convert('RGB')
    except Exception:
        draw.rectangle([0, 0, w + pad * 2, h + pad * 2], fill=(0, 0, 0))
        draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
        return im

EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
def find_matching_gt(hazy_path: str, gt_dir: str):
    hazy_name = os.path.basename(hazy_path)
    stem, ext = os.path.splitext(hazy_name)
    cand = os.path.join(gt_dir, hazy_name)
    if os.path.isfile(cand):
        return cand
    m = re.match(r'^(\d+)', stem)
    if m:
        base = m.group(1)
    else:
        base = stem.split('_')[0]
    for e in EXTS:
        cand = os.path.join(gt_dir, base + e)
        if os.path.isfile(cand):
            return cand
    files = []
    for e in ('*',):
        files.extend(glob.glob(os.path.join(gt_dir, f"{base}.{e}")))
    files = [f for f in files if os.path.isfile(f)]
    if files:
        def _score(p):
            _, e = os.path.splitext(p.lower())
            order = {'.png':0, '.jpg':1, '.jpeg':2, '.bmp':3}
            return order.get(e, 9)
        files.sort(key=_score)
        return files[0]
    return None

def forward_tiled_parallel(
    model, img,
    tile_size=512,
    tile_overlap=128,
    amp_inference=False,
    batch_tiles=0,
):
    B, C, H, W = img.shape
    assert B == 1, "Only support batch=1 input image"
    device = img.device
    tile_h = min(tile_size, H)
    tile_w = min(tile_size, W)
    overlap_h = min(tile_overlap, max(tile_h - 1, 0))
    overlap_w = min(tile_overlap, max(tile_w - 1, 0))
    step_h = max(1, tile_h - overlap_h)
    step_w = max(1, tile_w - overlap_w)
    top_vals = list(range(0, max(1, H - tile_h + 1), step_h))
    if top_vals[-1] + tile_h < H:
        top_vals.append(H - tile_h)
    left_vals = list(range(0, max(1, W - tile_w + 1), step_w))
    if left_vals[-1] + tile_w < W:
        left_vals.append(W - tile_w)
    coords, tiles = [], []
    for top in top_vals:
        for left in left_vals:
            bottom, right = top + tile_h, left + tile_w
            tiles.append(img[..., top:bottom, left:right])
            coords.append((top, left, bottom, right))
    out = torch.zeros_like(img)
    weight = torch.zeros((1, 1, H, W), device=device)
    if not batch_tiles or batch_tiles <= 0:
        batch_tiles = len(tiles)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_inference):
        for i in range(0, len(tiles), batch_tiles):
            batch = torch.cat(tiles[i:i + batch_tiles], dim=0)
            out_batch = model(batch).contiguous()
            for j in range(out_batch.size(0)):
                top, left, bottom, right = coords[i + j]
                rh, rw = bottom - top, right - left
                piece = _match_spatial(out_batch[j:j+1], rh, rw)
                out[..., top:bottom, left:right] += piece
                weight[..., top:bottom, left:right] += 1.0
    out = out / torch.clamp_min(weight, 1e-6)
    return out

def _match_spatial(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = t.shape
    if h == target_h and w == target_w:
        return t
    if h >= target_h and w >= target_w:
        sh = (h - target_h) // 2
        sw = (w - target_w) // 2
        return t[..., sh:sh+target_h, sw:sw+target_w]
    return F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)

def load_model(weights_path: str, device: torch.device, use_dp: bool = True) -> nn.Module:
    model = BVIFormer().to(device)
    if use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel).")
        model = nn.DataParallel(model)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    def add_module_prefix(sd):   return {f"module.{k}": v for k, v in sd.items()}
    def strip_module_prefix(sd): return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    want_module = isinstance(model, nn.DataParallel)
    has_module  = next(iter(state)).startswith("module.")
    if want_module and not has_module:
        state = add_module_prefix(state)
    if (not want_module) and has_module:
        state = strip_module_prefix(state)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"[load_model] Missing keys: {len(incompatible.missing_keys)} (show first 8)")
        for k in incompatible.missing_keys[:8]:
            print("  M:", k)
    if incompatible.unexpected_keys:
        print(f"[load_model] Unexpected keys: {len(incompatible.unexpected_keys)} (show first 8)")
        for k in incompatible.unexpected_keys[:8]:
            print("  U:", k)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazy_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--tile', type=int, default=336)
    parser.add_argument('--overlap', type=int, default=168)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--batch_tiles', type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dp = (device.type == 'cuda')
    model = load_model(args.weights, device, dp)
    os.makedirs(args.out_dir, exist_ok=True)
    hazy_list = []
    for e in EXTS:
        hazy_list += glob.glob(os.path.join(args.hazy_dir, f'*{e}'))
    hazy_list.sort()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    rows = [["name", "psnr", "ssim"]]
    with torch.no_grad():
        for hazy_path in hazy_list:
            name = os.path.basename(hazy_path)
            gt_path = find_matching_gt(hazy_path, args.gt_dir)
            if gt_path is None:
                print(f"[WARN] no GT:{name}, skipped.")
                continue
            hazy = Image.open(hazy_path).convert('RGB')
            gt   = Image.open(gt_path).convert('RGB')
            hazy_t = torch.from_numpy(np.asarray(hazy)).float().permute(2, 0, 1) / 255.0
            gt_t   = torch.from_numpy(np.asarray(gt)).float().permute(2, 0, 1) / 255.0
            H = min(hazy_t.shape[1], gt_t.shape[1])
            W = min(hazy_t.shape[2], gt_t.shape[2])
            hazy_t = hazy_t[:, :H, :W].unsqueeze(0).to(device, non_blocking=True)
            gt_t   = gt_t[:, :H, :W].to(device, non_blocking=True)
            out = forward_tiled_parallel(
                model, hazy_t,
                tile_size=args.tile, tile_overlap=args.overlap,
                amp_inference=args.bf16, batch_tiles=args.batch_tiles
            ).clamp_(0, 1)
            ps = compute_psnr(out[0], gt_t)
            ss = compute_ssim(out[0], gt_t)
            total_psnr += ps
            total_ssim += ss
            count += 1
            out_im = Image.fromarray(tensor_to_uint8_rgb(out[0]))
            out_im = draw_text(out_im, f'PSNR {ps:.2f}   SSIM {ss:.4f}')
            save_path = os.path.join(args.out_dir, os.path.splitext(name)[0] + '.png')
            out_im.save(save_path)
            rows.append([name, f"{ps:.4f}", f"{ss:.6f}"])

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        if count > 0:
            writer.writerow(["mean", f"{total_psnr / count:.4f}", f"{total_ssim / count:.6f}"])
    print(f"Done. Images={count}, mean PSNR={total_psnr / max(count,1):.3f}, mean SSIM={total_ssim / max(count,1):.5f}")
    print(f"Saved to: {args.out_dir}")
    print(f"CSV: {args.output_csv}")

if __name__ == "__main__":
    main()
