# im2im
# Copyright (C) 2023  Lopho <contact@lopho.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from PIL import Image

from unet import UNet
from unet_v2 import UNetV2


def load_batch(batch: list[str]) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack([ torch.tensor(np.asarray(Image.open(f))) for f in batch ]).permute(0,3,1,2)
    images = images.to(torch.float32).div(255).to(memory_format = torch.contiguous_format)
    return { 'images': images, 'file_names': batch }

def save_batch(batch: torch.Tensor, file_names: list[str], output_dir: str) -> None:
    batch = batch.mul(255).round().clamp(0,255).permute(0,2,3,1).to(device = 'cpu', dtype = torch.uint8).numpy()
    for i,x in enumerate(batch):
        f = os.path.join(output_dir, os.path.basename(file_names[i]))
        Image.fromarray(x).save(f)


def load_model(path: str, device: str | torch.device = 'cuda', v2: bool = False) -> UNet:
    with open(os.path.join(path, 'configuration.json'), 'r') as f:
        config = json.load(f)
    if v2:
        model = UNetV2(**config['config'])
    else:
        model = UNet(**config['config'])
    model.load_state_dict(torch.load(os.path.join(path, config['ckpt'])))
    model = model.eval().requires_grad_(False).to(device = device, memory_format = torch.contiguous_format)
    model = torch.compile(model, mode = 'max-autotune', fullgraph = True)
    return model


@torch.inference_mode()
def process_images(
        model: UNet | UNetV2,
        input_dir: str,
        output_dir: str,
        batch_size: int = 16,
        device: str | torch.device = 'cuda',
        num_workers: int = 2,
        **_
) -> None:
    device = torch.device(device)
    dataset = sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir))
    dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            collate_fn = load_batch,
            num_workers = num_workers,
            pin_memory = device.type != 'cpu',
            pin_memory_device = str(device) if device.type != 'cpu' else ''
    )
    pbar = tqdm(total = len(dataset), dynamic_ncols = True, smoothing = 1)
    for b in dataloader:
        x = b['images'].to(device = device, non_blocking = True)
        bs,_,h,w = x.shape
        right = math.ceil(w / model.ksize) * model.ksize - w
        bottom = math.ceil(h / model.ksize) * model.ksize - h
        x = torch.nn.functional.pad(x, [0, right, 0, bottom], mode = 'reflect')
        with torch.autocast(device.type):
            y = model(x)
        y = y[:,:,0:h,0:w]
        save_batch(y, b['file_names'], output_dir)
        pbar.update(bs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type = str, required = True)
    parser.add_argument('--output_dir', '-o', type = str, required = True)
    parser.add_argument('--checkpoint', '-p', type = str, required = True)
    parser.add_argument('--device', '-d', type = str, default = 'cuda')
    parser.add_argument('--batch_size', '-b', type = int, default = 16)
    parser.add_argument('--num_workers', '-w', type = int, default = 2)
    parser.add_argument('--v2', type = bool, action = 'store_true')
    args = parser.parse_args()
    print({**vars(args)})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    model = load_model(args.checkpoint, device = args.device, v2 = args.v2)
    process_images(model = model, **vars(args))

