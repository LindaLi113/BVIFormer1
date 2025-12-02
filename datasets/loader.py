import os, random, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_files(dir_or_list):
    dirs = dir_or_list if isinstance(dir_or_list, (list, tuple)) else dir_or_list.split(";")
    files = []
    for d in dirs:
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(IMG_EXT)]
    return files

def _open(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.

def _rand_crop(h, c, ps):
    if ps <= 0:
        return h, c
    H, W, _ = h.shape
    if H < ps or W < ps:
        ph, pw = max(0, ps - H), max(0, ps - W)
        h = np.pad(h, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        c = np.pad(c, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        H, W = h.shape[:2]
    top, left = random.randint(0, H - ps), random.randint(0, W - ps)
    return h[top:top+ps, left:left+ps], c[top:top+ps, left:left+ps]

class UseDataset(Dataset):
    def __init__(self,
                 hazy_dir,
                 clear_dir,
                 train: bool = True,
                 patch_size: int = 256,
                 augment: bool = False):
        self.train = train
        self.ps    = patch_size
        self.aug   = augment
        hazy  = _list_files(hazy_dir)
        clear = _list_files(clear_dir)
        cmap  = {os.path.splitext(os.path.basename(p))[0]: p for p in clear}
        self.hazy_paths, self.clear_paths = [], []
        miss = 0
        for hp in hazy:
            base = os.path.splitext(os.path.basename(hp))[0].split("_")[0]
            cp   = cmap.get(base)
            if cp:
                self.hazy_paths.append(hp)
                self.clear_paths.append(cp)
            else:
                miss += 1
        print(f"[Dataset] paired {len(self.hazy_paths)} imgs, drop {miss} hazy")

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        h = _open(self.hazy_paths[idx])
        c = _open(self.clear_paths[idx])
        if self.train:
            h, c = _rand_crop(h, c, self.ps)
            if self.aug:
                if random.random() < .5:
                    h = h[:, ::-1].copy()
                    c = c[:, ::-1].copy()
                if random.random() < .5:
                    h = h[::-1, :].copy()
                    c = c[::-1, :].copy()
        h = np.ascontiguousarray(h)
        c = np.ascontiguousarray(c)
        h = torch.from_numpy(h.transpose(2, 0, 1)).float()
        c = torch.from_numpy(c.transpose(2, 0, 1)).float()
        return h, c
