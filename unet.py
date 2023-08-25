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

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 3, padding = 'same', padding_mode = 'reflect')
        self.norm = nn.GroupNorm(1, channels_out) if norm else nn.Identity()
        self.act = nn.PReLU(channels_out) if prelu else nn.ReLU()
        self.residual = residual and channels_out == channels_in
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            x0 = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.residual:
            x = x0 + x
        return x

class InBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            dropout: float,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_out, norm, prelu, residual)
        self.conv2 = ConvBlock(channels_out, channels_out, norm, prelu, residual)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

class OutBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_mid: int,
            channels_out: int,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_mid * 2, norm, prelu, residual)
        self.conv2 = ConvBlock(channels_mid * 2, channels_mid, norm, prelu, residual)
        self.conv_out = nn.Conv2d(channels_mid, channels_out, 3, padding = 'same', padding_mode = 'reflect')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_out(x)
        return x

class DownBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            dropout: float,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(channels_in, channels_out, norm, prelu, residual)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            dropout: float,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_out, norm, prelu, residual)
        self.conv2 = ConvBlock(channels_out, channels_out, norm, prelu, residual)
        self.up = nn.ConvTranspose2d(channels_out, channels_out, 3, stride = 2, padding = 1, output_padding = 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.drop(x)
        return x

class MidBlock(nn.Module):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            dropout: float,
            norm: bool = True,
            prelu: bool = True,
            residual: bool = False
    ) -> None:
        super().__init__()
        self.down = DownBlock(channels_in, channels_out, dropout, norm, prelu, residual)
        self.conv = ConvBlock(channels_out, channels_out, norm, prelu, residual)
        self.up = nn.ConvTranspose2d(channels_out, channels_out, 3, stride = 2, padding = 1, output_padding = 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        x = self.up(x)
        x = self.drop(x)
        return x

class UNet(nn.Module):
    def __init__(self,
            blocks: int = 6,
            channels_in: int = 3,
            channels_out: int = 3,
            dropout: float = 0,
            norm: bool = False,
            prelu: bool = False,
            residual: bool = True,
    ) -> None:
        super().__init__()
        self.ksize: int = 2 ** blocks
        self.drop_in = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down_blocks: nn.ModuleList[DownBlock] = nn.ModuleList((
                InBlock(channels_in, 48, dropout, norm, prelu, residual),
                *( DownBlock(48, 48, dropout, norm, prelu, residual) for _ in range(blocks - 1) )
        ))
        self.mid_block = MidBlock(48, 48, dropout, norm, prelu, residual)
        self.up_blocks: nn.ModuleList[UpBlock] = nn.ModuleList((
                *( UpBlock(144 if i > 0 else 96, 96, dropout, norm, prelu, residual) for i in range(blocks) ),
                OutBlock(96 + channels_in, 32, channels_out, norm, prelu, residual)
        ))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(x)
        pools: list[torch.Tensor] = [ x ]
        for l in self.down_blocks:
            x = l(x)
            pools.append(x)
        x = self.mid_block(x)
        for l in self.up_blocks:
            x = torch.cat([ x, pools.pop() ], dim = 1)
            x = l(x)
        return x

