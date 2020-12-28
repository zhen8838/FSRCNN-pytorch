import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from models import FSRCNN
import torch
import numpy as np
import cv2
from fast_run_utils import *
from pathlib import Path
import time
import argparse


def mask_thresh(splited_image, hw, metric, threshold) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  # convert splited_image [h,w,...] to [h*w,...] for fast indexing
  h, w = hw
  splited_image = splited_image.reshape((h * w, *splited_image.shape[2:]))
  metric = torch.flatten(metric)
  true_idx = torch.flatten(torch.nonzero(metric > threshold, as_tuple=False))
  false_idx = torch.flatten(torch.nonzero(~(metric > threshold), as_tuple=False))

  true_im = splited_image[true_idx]
  false_im = splited_image[false_idx]
  return true_idx, true_im, false_idx, false_im


def mask_inverse(true_idx: torch.Tensor, true_im: torch.Tensor, false_idx, false_im, hw) -> torch.Tensor:
  if true_im.numel() == 0:
    splited_im = false_im
    idx = false_idx
  elif false_im.numel() == 0:
    splited_im = true_im
    idx = true_idx
  else:
    splited_im = torch.cat((true_im, false_im), 0)
    idx = torch.cat((true_idx, false_idx), 0)

  splited_im = splited_im[torch.argsort(idx)]
  return splited_im.view((*hw, *splited_im.shape[1:]))


def main(video='/home/zqh/Videos/newland.flv', weights='./fsrcnn_x2.pth',
         threshold=120, stride=40, scale=2,
         orginal_method: bool = False):

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = FSRCNN(2)
  model.load_state_dict(torch.load(weights))
  model = model.eval().to(device)

  ptv = PatchTotalVariation()
  g, length, fps, height, width = get_read_stream(Path(video))
  plt.ion()
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9))
  title = plt.title('fps:')
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1im = ax1.imshow(np.zeros((height, width, 3)))
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2im = ax2.imshow(np.zeros((height // stride, width // stride)), vmin=0, vmax=255)
  ax3.set_xticks([])
  ax3.set_yticks([])
  ax3im = ax3.imshow(np.zeros((height * scale, width * scale, 3)))
  plt.tight_layout()
  plt.show()
  for im in g:
    orginal_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if orginal_method:
      hr_rgb = cv2.resize(orginal_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      ycrcb = cv2.cvtColor(orginal_rgb, cv2.COLOR_RGB2YCrCb)
      hr_ycrcb = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2YCrCb)
      im = torch.ByteTensor(ycrcb[..., 0:1]).to(device)
    else:
      im = torch.ByteTensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(device)
    channel = im.shape[-1]

    start_time = time.time()
    # F.interpolate(im.permute((2, 1, 0))[None,...], scale_factor=0.5)[0].permute((1, 2, 0))
    split_im, hw = window_split(im, stride)
    split_tv = ptv(split_im)
    true_idx, true_im, false_idx, false_im = mask_thresh(split_im, hw, split_tv, threshold)

    # NOTE this model only accpect channel==1, so need reshape
    if true_im.numel() > 0:
      true_im = (model(true_im.reshape(-1, 1, *true_im.shape[2:]) / 255.) * 255.)
      true_im = true_im.byte().reshape((-1, channel, *true_im.shape[2:]))
    # fast interpolate for false_im
    if false_im.numel() > 0:
      false_im = F.interpolate(false_im.float(),
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=False).byte()

    # merge image
    processed_im = mask_inverse(true_idx, true_im, false_idx, false_im, hw)

    new_im = window_merge(processed_im, hw, stride, scale)

    ax1im.set_data(orginal_rgb)
    ax2im.set_data(split_tv.detach().to('cpu').numpy())

    if orginal_method:
      new_ycrcb = np.concatenate((new_im.detach().to('cpu').numpy(), hr_ycrcb[..., 1:]), -1)
      ax3im.set_data(cv2.cvtColor(new_ycrcb, cv2.COLOR_YCrCb2RGB))
    else:
      ax3im.set_data(new_im.detach().to('cpu').numpy())
    title.set_text(f'fps: {1.0 / (time.time() - start_time):.3f}')
    # plt.pcolor
    fig.canvas.flush_events()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, default='/home/zqh/Videos/newland.flv')
  parser.add_argument("--weights", type=str, default='./fsrcnn_x2.pth')
  parser.add_argument("--threshold", type=int, default=120)
  parser.add_argument("--stride", type=int, default=40)
  parser.add_argument("--scale", type=int, default=2)
  parser.add_argument("--orginal_method", action="store_true", help="add --orginal_method to enable orginal forward mode")
  args = parser.parse_args()
  main(args.video, args.weights, args.threshold, args.stride, args.scale, args.orginal_method)
