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

from typing import Any
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 3, padding = 'same', padding_mode = 'reflect')
        self.act = nn.LeakyReLU(0.1)
        self.skip = channels_in == channels_out
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip:
            x_0 = x
        x = self.conv(x)
        x = self.act(x)
        if self.skip:
            x = x + x_0
        return x


class InBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(channels_in, channels_out)
        self.down = DownBlock(channels_out, channels_out)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.down(x)
        return x


class OutBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_mid: int,
            channels_out: int
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_mid)
        self.conv2 = ConvBlock(channels_mid, channels_mid // 2)
        self.conv_out = nn.Conv2d(channels_mid // 2, channels_out, 3, padding = 'same', padding_mode = 'reflect')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_out(x)
        return x


class DownBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(channels_in, channels_out)
        self.pool = nn.Conv2d(channels_out, channels_out, 3, stride = 2, padding = 1, padding_mode = 'reflect')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_out)
        self.conv2 = ConvBlock(channels_out, channels_out)
        self.up = nn.ConvTranspose2d(channels_out, channels_out, 3, stride = 2, padding = 1, output_padding = 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x


class MidBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int
    ) -> None:
        super().__init__()
        self.down = DownBlock(channels_in, channels_out)
        self.conv = ConvBlock(channels_out, channels_out)
        self.up = nn.ConvTranspose2d(channels_out, channels_out, 3, stride = 2, padding = 1, output_padding = 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self,
            channels_in: int = 3,
            channels_out: int = 3,
            blocks: int = 6,
            stepping: tuple[int, int] = (48, 96, 64)
    ) -> None:
        super().__init__()
        self.ksize: int = 2 ** blocks
        self.down_blocks: nn.ModuleList[DownBlock] = nn.ModuleList((
                InBlock(channels_in, stepping[0]),
                *( DownBlock(stepping[0], stepping[0]) for _ in range(blocks - 2) )
        ))
        self.mid_block = MidBlock(stepping[0], stepping[0])
        self.up_blocks: nn.ModuleList[UpBlock] = nn.ModuleList((
                *( UpBlock(stepping[0] + stepping[1] if i > 0 else stepping[1], stepping[1]) for i in range(blocks - 1) ),
                OutBlock(stepping[1] + channels_in, stepping[2], channels_out)
        ))
        self.hparams: dict[str, Any] = {
            'blocks': blocks,
            'fixed_step': True,
            'stepping': stepping,
            'net': self.__class__.__name__
        }
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pools: list[torch.Tensor] = [ x ]
        for l in self.down_blocks:
            x = l(x)
            pools.append(x)
        x = self.mid_block(x)
        for l in self.up_blocks:
            x = torch.cat([ x, pools.pop() ], dim = 1)
            x = l(x)
        return x

