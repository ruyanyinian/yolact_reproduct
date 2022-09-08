# -*- coding: utf-8 -*-
import torch
from itertools import product
from math import sqrt
import numpy as np


def box_iou(box_a, box_b):
  """
  Compute the IoU of two sets of boxes.
  Args:
      box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
      box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
  Return:
      jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
  """
  use_batch = True
  if box_a.dim() == 2:
    use_batch = False
    box_a = box_a[None, ...]
    box_b = box_b[None, ...]

  (n, A), B = box_a.shape[:2], box_b.shape[1]
  # add a dimension
  box_a = box_a[:, :, None, :].expand(n, A, B, 4)
  box_b = box_b[:, None, :, :].expand(n, A, B, 4)

  max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
  min_xy = torch.max(box_a[..., :2], box_b[..., :2])
  inter = torch.clamp((max_xy - min_xy), min=0)
  inter_area = inter[..., 0] * inter[..., 1]

  area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
  area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

  out = inter_area / (area_a + area_b - inter_area)
  return out if use_batch else out.squeeze(0)


def box_iou_numpy(box_a, box_b):
  (n, A), B = box_a.shape[:2], box_b.shape[1]
  # add a dimension
  box_a = np.tile(box_a[:, :, None, :], (1, 1, B, 1))
  box_b = np.tile(box_b[:, None, :, :], (1, A, 1, 1))

  max_xy = np.minimum(box_a[..., 2:], box_b[..., 2:])
  min_xy = np.maximum(box_a[..., :2], box_b[..., :2])
  inter = np.clip((max_xy - min_xy), a_min=0, a_max=100000)
  inter_area = inter[..., 0] * inter[..., 1]

  area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
  area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

  return inter_area / (area_a + area_b - inter_area)


def match(cfg, box_gt, anchors, class_gt):
  # 将18525个anchor框转换成 [xmin, ymin, xmax, ymax].
  decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
  # box_gt是来自于annotation里面的。当前图像的annotation的target有两个, 所以box_gt=(2,4)
  overlaps = box_iou(box_gt, decoded_priors)  # 对每一个anchor对gt_annotation做iOU计算,最后的shape是(2, 18525)

  # 从18525个anchor框挑选出两个anchor框,这两个anchor框是和gt annotation做的最大的iOU值。gt_max_i=tensor([18355, 15863])
  _, gt_max_i = overlaps.max(1)
  # overlaps.max(0):对每一个anchor来说, 这两个gt annotation中,哪一个gt annotation和当前的anchor产生最大的iOU
  # each_anchor_max就是具体的值是多少,而anchor_max_i是告诉你具体的哪一个GT annotation和当前的anchor产生最大的iOU
  each_anchor_max, anchor_max_i = overlaps.max(0)  # (num_achors, ) the max IoU for each anchor

  # For the max IoU anchor for each gt box, set its IoU to 2. This ensures that it won't be filtered
  # in the threshold step even if the IoU is under the negative threshold. This is because that we want
  # at least one anchor to match with each gt box or else we'd be wasting training data.
  # 我们把最大的anchor框的iOU设置成2,以免被过滤
  each_anchor_max.index_fill_(0, gt_max_i, 2)

  # Set the index of the pair (anchor, gt) we set the overlap for above. 也就是把anchor和gt设置成一个pair
  for j in range(gt_max_i.size(0)):
    anchor_max_i[gt_max_i[j]] = j
  # 每一个anchor框存储了最大的GT annotation
  anchor_max_gt = box_gt[anchor_max_i]  # (18525, 4)
  # 在这里开始对框进行了正负样本的分配。分别分配正样本, 中立样本和负样本
  conf = class_gt[anchor_max_i] + 1  # the class of the max IoU gt box for each anchor
  conf[each_anchor_max < cfg.pos_iou_thre] = -1  # label as neutral
  conf[each_anchor_max < cfg.neg_iou_thre] = 0  # label as background

  offsets = encode(anchor_max_gt, anchors) # 把anchor对应的GT annotation的具体值给进行offset操作, 结果是(18525,4)

  return offsets, conf, anchor_max_gt, anchor_max_i


def make_anchors(cfg, conv_h, conv_w, scale):
  # 这个是对每一个特征点铺设9个anchor, 铺设anchor的操作类似于
  prior_data = []
  # Iteration order is important (it has to sync up with the convout)
  for j, i in product(range(conv_h), range(conv_w)):  # 这里i相当与遍历w, j相当于遍历h
    # + 0.5 because priors are in center, 这个遍历是center点在FPN特征图下的坐标
    x = (i + 0.5) / conv_w
    y = (j + 0.5) / conv_h

    for ar in cfg.aspect_ratios: # cfg.aspect_ratios=[1,0.5,2]
      ar = sqrt(ar)
      w = scale * ar / cfg.img_size
      h = scale / ar / cfg.img_size

      prior_data += [x, y, w, h]

  return prior_data


def encode(matched, priors):
  variances = [0.1, 0.2]

  g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 10 * (Xg - Xa) / Wa
  g_cxcy /= (variances[0] * priors[:, 2:])  # 10 * (Yg - Ya) / Ha
  g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 5 * log(Wg / Wa)
  g_wh = torch.log(g_wh) / variances[1]  # 5 * log(Hg / Ha)
  # return target for smooth_l1_loss
  offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]

  return offsets


def sanitize_coordinates(_x1, _x2, img_size, padding=0):
  """
  Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
  Also converts from relative to absolute coordinates and casts the results to long tensors.

  Warning: this does things in-place behind the scenes so copy if necessary.
  """
  _x1 = _x1 * img_size
  _x2 = _x2 * img_size

  x1 = torch.min(_x1, _x2)
  x2 = torch.max(_x1, _x2)
  x1 = torch.clamp(x1 - padding, min=0)
  x2 = torch.clamp(x2 + padding, max=img_size)

  return x1, x2


def sanitize_coordinates_numpy(_x1, _x2, img_size, padding=0):
  _x1 = _x1 * img_size
  _x2 = _x2 * img_size

  x1 = np.minimum(_x1, _x2)
  x2 = np.maximum(_x1, _x2)
  x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
  x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)

  return x1, x2


def crop(masks, boxes, padding=1):
  """
  "Crop" predicted masks by zeroing out everything not in the predicted bbox.
  Args:
      - masks should be a size [h, w, n] tensor of masks
      - boxes should be a size [n, 4] tensor of bbox coords in relative point form
  """
  h, w, n = masks.size()  # 136, 136, n=2
  # 对9个正样本的anchor框的相对路径转换成(136,136)下的绝对路径
  x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
  y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

  rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
  cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

  masks_left = rows >= x1.view(1, 1, -1)
  masks_right = rows < x2.view(1, 1, -1)
  masks_up = cols >= y1.view(1, 1, -1)
  masks_down = cols < y2.view(1, 1, -1)

  crop_mask = masks_left * masks_right * masks_up * masks_down  # 把利用乘法把mask给crop下来

  return masks * crop_mask.float() # 把crop下来的mask贴合到图里面


def crop_numpy(masks, boxes, padding=1):
  h, w, n = masks.shape
  x1, x2 = sanitize_coordinates_numpy(boxes[:, 0], boxes[:, 2], w, padding)
  y1, y2 = sanitize_coordinates_numpy(boxes[:, 1], boxes[:, 3], h, padding)

  rows = np.tile(np.arange(w)[None, :, None], (h, 1, n))
  cols = np.tile(np.arange(h)[:, None, None], (1, w, n))

  masks_left = rows >= (x1.reshape(1, 1, -1))
  masks_right = rows < (x2.reshape(1, 1, -1))
  masks_up = cols >= (y1.reshape(1, 1, -1))
  masks_down = cols < (y2.reshape(1, 1, -1))

  crop_mask = masks_left * masks_right * masks_up * masks_down

  return masks * crop_mask


def mask_iou(mask1, mask2):
  """
  Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
  Note: if iscrowd is True, then mask2 should be the crowd.
  """
  intersection = torch.matmul(mask1, mask2.t())
  area1 = torch.sum(mask1, dim=1).reshape(1, -1)
  area2 = torch.sum(mask2, dim=1).reshape(1, -1)
  union = (area1.t() + area2) - intersection
  ret = intersection / union

  return ret.cpu()
