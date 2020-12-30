import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from models import FSRCNN
import torch
import numpy as np
import cv2
import PIL.Image as pil_image
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb
from fast_run_utils import *
from pathlib import Path
import time
import argparse


def mask_thresh(splited_image, hw, boolean) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  # convert splited_image [h,w,...] to [h*w,...] for fast indexing
  h, w = hw
  splited_image = splited_image.reshape((h * w, *splited_image.shape[2:]))
  boolean = boolean.flatten()
  true_idx = torch.flatten(torch.nonzero(boolean, as_tuple=False))
  false_idx = torch.flatten(torch.nonzero(torch.logical_not(boolean), as_tuple=False))
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
         threshold=0.5, stride=40, scale=2,
         orginal_method: bool = False,
         export: bool = False):

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = FSRCNN(2)
  model.load_state_dict(torch.load(weights))
  model = model.eval().to(device)

  ptv = PatchTotalVariation()
  video = Path(video)
  g, length, fps, height, width = get_read_stream(video)
  print(f"Video info: \nheight: {height} width:{width} length:{length} fps:{fps}")
  if export:
    video_export = video.parent / (video.stem + '_out' + '.mp4')
    writer = get_writer_stream(video_export, fps, height * scale, width * scale)

  plt.ion()
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9))
  title = plt.title('fps:')
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1im = ax1.imshow(np.zeros((height, width, 3)))
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2im = ax2.imshow(np.zeros((height // stride, width // stride)), vmin=0, vmax=1)
  ax3.set_xticks([])
  ax3.set_yticks([])
  ax3im = ax3.imshow(np.zeros((height * scale, width * scale, 3)))
  plt.tight_layout()
  plt.show()
  for im in g:
    orginal_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if orginal_method:
      hr_rgb = pil_image.fromarray(orginal_rgb, mode='RGB').resize(
          (width * scale, height * scale), resample=pil_image.BICUBIC)
      ycbcr = convert_rgb_to_ycbcr(orginal_rgb.astype(np.float32))
      hr_ycbcr = convert_rgb_to_ycbcr(np.array(hr_rgb).astype(np.float32))
      im = torch.tensor(ycbcr[..., 0:1],
                        dtype=torch.float32, device=device)
    else:
      im = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), dtype=torch.float32, device=device)
    channel = im.shape[-1]

    start_time = time.time()
    # F.interpolate(im.permute((2, 1, 0))[None,...], scale_factor=0.5)[0].permute((1, 2, 0))
    split_im, hw = window_split(im, stride)
    split_tv = ptv(split_im)
    #  NOTE nromlize tv value to [0,1] and set patch color==1 when it's tv>threshold
    split_tv.div_(split_tv.max())
    boolean = split_tv > threshold
    true_idx, true_im, false_idx, false_im = mask_thresh(split_im, hw, boolean)
    split_tv[boolean] = 1

    # NOTE this model only accpect channel==1, so need reshape
    if true_im.numel() > 0:
      true_im = (model(true_im.reshape(-1, 1, *true_im.shape[2:]) / 255.).clamp(0.0, 1.0) * 255.)
      true_im = true_im.byte().reshape((-1, channel, *true_im.shape[2:]))
    # fast interpolate for false_im
    if false_im.numel() > 0:
      false_im = F.interpolate(false_im,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
    # merge image
    processed_im = mask_inverse(true_idx, true_im, false_idx, false_im, hw)

    new_im = window_merge(processed_im, hw, stride, scale)

    ax1im.set_data(orginal_rgb)
    ax2im.set_data(split_tv.detach().to('cpu').numpy())

    if orginal_method:
      new_ycbcr = np.concatenate(
          (new_im.detach().to('cpu').numpy(), hr_ycbcr[..., 1:]), -1)
      new_rgb = np.clip(convert_ycbcr_to_rgb(new_ycbcr), 0., 255.).astype('uint8')
    else:
      new_rgb = new_im.detach().to('cpu').numpy().astype('uint8')
    ax3im.set_data(new_rgb)
    title.set_text(f'fps: {1.0 / (time.time() - start_time):.3f}')
    if export:
      writer.write(cv2.cvtColor(new_rgb, cv2.COLOR_RGB2BGR))
    # plt.pcolor
    fig.canvas.flush_events()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, default='/home/zqh/Videos/newland.flv')
  parser.add_argument("--weights", type=str, default='./fsrcnn_x2.pth')
  parser.add_argument("--threshold", type=int, default=0.5)
  parser.add_argument("--stride", type=int, default=40)
  parser.add_argument("--scale", type=int, default=2)
  parser.add_argument("--orginal_method", action="store_true",
                      help="add --orginal_method to enable orginal forward mode")
  parser.add_argument("--export", action="store_true",
                      help="add --export to enable save result video, then you can use `ffmpeg -i xx.mp4 -vcodec libx265 -crf 28 xxx.mp4` to compress viedo")
  args = parser.parse_args()
  main(args.video, args.weights, args.threshold, args.stride,
       args.scale, args.orginal_method, args.export)
