import torch
import numpy as np
import cv2

from typing import Tuple, Generator
from pathlib import Path
import torch.nn.functional as F


def imread(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def measure(img, measure_fuc):
  return measure_fuc(img)


def window_split(image: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
  """ split sub image by stride

  Args:
      image (torch.Tensor): shape [h,w,c]
      stride (int): must can be divide by h and w

  Returns:
      [torch.Tensor,torch.Tensor]:
      splited image: shape [h//stride,w//stride,c,stride,stride]
      new hw: [h//stride,w//stride]
  """
  splited_image = image.unfold(0, stride, stride).unfold(1, stride, stride)
  return splited_image, splited_image.shape[:2]


def window_merge(image: torch.Tensor, hw, stride, scale):
  h, w = hw
  image = image.permute((0, 3, 1, 4, 2))
  image = image.reshape((h * stride * scale, w * stride * scale, image.shape[-1]))
  return image


class PatchTotalVariation(object):
  @staticmethod
  def totalvariation(image: torch.Tensor, ksize=1):
    """total variation, return ||total variation||1
    NOTE: l1 norm : ||y||=|y_1|+|y_2|, total variation= sum(x_i_j)
    Args:
        image (torch.Tensor): [...,c,h,w] , type: float32
        ksize (int, optional): Defaults to 1.

    Returns:
        torch.Tensor: ||total variation||1, shape [h,w]
    """
    dh = image[..., ksize:, :] - image[..., :-ksize, :]
    dw = image[..., :, ksize:] - image[..., :, :-ksize]
    dh = F.pad(dh, [0, 0, 0, ksize], mode='constant', value=0)
    dw = F.pad(dw, [0, ksize], mode='constant', value=0)
    tv = torch.sum(torch.abs(dh) + torch.abs(dw), (-3, -2, -1))
    return tv

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    return self.totalvariation(image)


def get_read_stream(path: Path) -> Tuple[Generator, int, int, int, int]:
  read_stream = cv2.VideoCapture(path.as_posix())
  length = int(read_stream.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(read_stream.get(cv2.CAP_PROP_FPS))
  height = int(read_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(read_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

  def gen(stream):
    while True:
      ret, frame = stream.read()
      if ret:
        yield frame
      else:
        stream.release()
        break
  return gen(read_stream), length, fps, height, width

def get_writer_stream(path: Path, fps: int, height: int, width: int):
  writer_stream = cv2.VideoWriter(
      filename=path.as_posix(),
      fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
      fps=fps,
      frameSize=(width, height))
  return writer_stream
