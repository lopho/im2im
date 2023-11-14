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
import mediapy
from concurrent.futures import ThreadPoolExecutor, Future
from functools import partial

from unet import UNet
from unet_v2 import UNetV2

import gc

def chunk(lst, n: int) -> list:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

@torch.inference_mode()
def _load_batch(batch: list[Image.Image]) -> torch.Tensor:
    images = torch.stack([ torch.tensor(np.asarray(x)) for x in batch ]).permute(0,3,1,2)
    images = images.to(torch.float32).div(255).to(memory_format = torch.contiguous_format)
    return images

@torch.inference_mode()
def _save_batch(batch: torch.Tensor) -> np.ndarray:
    batch = batch.mul(255).round().clamp(0,255).to(dtype = torch.uint8).permute(0,2,3,1)
    return batch

@torch.inference_mode()
def load_video_as_batch(batch: list[str], ksize: int, bsize: int) -> dict[str, torch.Tensor | list[str]]:
    # only batch size 1 accepted, since it returns all frames of the video as a batch
    assert len(batch) == 1
    video_path = batch[0]
    frames = mediapy.read_video(video_path)
    fps = frames.metadata.fps
    images = _load_batch(frames)
    _,_,h,w = images.shape
    right = math.ceil(w / model.ksize) * model.ksize - w
    bottom = math.ceil(h / model.ksize) * model.ksize - h
    xs = []
    f = len(images)
    for c in chunk(images, bsize):
        c = torch.nn.functional.pad(c, [0, right, 0, bottom], mode = 'reflect').to(
                dtype = torch.float16,
                memory_format = torch.channels_last
        )
        if len(c) < bsize:
            c = torch.stack([*c] * math.ceil(bsize / len(c)))[0:bsize]
        xs.append(c)
    xs = torch.cat(xs)
    del images
    gc.collect()
    return { 'images': xs, 'file_name': os.path.basename(video_path), 'f': f, 'w': w, 'h': h, 'fps': fps }

@torch.inference_mode()
def save_batch_as_video(batch: torch.Tensor, file_name: str, output_dir: str, fps: float) -> None:
    mediapy.write_video(os.path.join(output_dir, file_name), batch, fps=fps, codec='vp9', qp=15)

@torch.inference_mode()
def load_batch(batch: list[str]) -> dict[str, torch.Tensor | list[str]]:
    images = [ Image.open(f) for f in batch ]
    images = _load_batch(images)
    return { 'images': images, 'file_names': batch }


def save_batch(batch: torch.Tensor, file_names: list[str], output_dir: str) -> None:
    batch = _save_batch(batch).to(device = 'cpu').numpy()
    for i,x in enumerate(batch):
        f = os.path.join(output_dir, os.path.basename(file_names[i]))
        Image.fromarray(x).save(f)


def load_model(path: str, device: str | torch.device = 'cuda', v2: bool = False, compile: bool = False) -> UNet:
    with open(os.path.join(path, 'configuration.json'), 'r') as f:
        config = json.load(f)
    if v2:
        model = UNetV2(**config['config'])
    else:
        model = UNet(**config['config'])
    model.load_state_dict(torch.load(os.path.join(path, config['ckpt'])))
    model = model.eval().requires_grad_(False).to(
            device = device,
            dtype = torch.float16,
            memory_format = torch.channels_last
    )
    if compile:
        model = torch.compile(model)
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
        save_batch(y, b['file_name'], output_dir)
        pbar.update(bs)

class MockFuture:
    def result(self):
        return None

@torch.inference_mode()
def process_videos(
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
            batch_size = 1,
            shuffle = False,
            drop_last = False,
            collate_fn = partial(load_video_as_batch,
                    ksize = model.ksize,
                    bsize = batch_size
            ),
            num_workers = num_workers
    )
    pbar = tqdm(total = len(dataset), dynamic_ncols = True, position = 1, leave = True, smoothing = 0.01)
    pool = ThreadPoolExecutor(8)
    futures: list[Future] = []
    def build(x,h,w,f):
        ys = []
        for c in chunk(x, batch_size):
            c = c.to(device = device)
            y = model(c)[:,:,0:h,0:w]
            y = _save_batch(y).to(device = 'cpu')
            ys.append(y)
        ys = torch.cat(ys)[0:f]
        return ys
    build = torch.compile(build)
    for b in dataloader:
        ys = build(b['images'], b['h'], b['w'], b['f'])
        #ys = _save_batch(ys)
        futures.append(pool.submit(save_batch_as_video, ys.numpy(), b['file_name'], output_dir, b['fps']))
        while len(futures) > 16:
            for i in range(len(futures)):
                if futures[i].done():
                    futures.pop(i).result()
                    break
        pbar.update(1)
    pool.shutdown(wait = True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type = str, required = True)
    parser.add_argument('--output_dir', '-o', type = str, required = True)
    parser.add_argument('--checkpoint', '-p', type = str, required = True)
    parser.add_argument('--device', '-d', type = str, default = 'cuda')
    parser.add_argument('--batch_size', '-b', type = int, default = 16)
    parser.add_argument('--num_workers', '-w', type = int, default = 2)
    parser.add_argument('--compile', action = 'store_true')
    parser.add_argument('--v2', action = 'store_true')
    args = parser.parse_args()
    print({**vars(args)})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    model = load_model(args.checkpoint, device = args.device, v2 = args.v2, compile = args.compile)
    os.makedirs(args.output_dir, exist_ok = True)
    process_videos(model = model, **vars(args))

