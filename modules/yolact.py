import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.resnet import ResNet
from utils.box_utils import match, crop, make_anchors
from modules.swin_transformer import SwinTransformer
import pdb


class PredictionModule(nn.Module):
  def __init__(self, cfg, coef_dim=32):
    super().__init__()
    self.num_classes = cfg.num_classes
    self.coef_dim = coef_dim

    self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
    self.bbox_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
    self.conf_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
    self.coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                              kernel_size=3, padding=1),
                                    nn.Tanh())

  def forward(self, x):
    x = self.upfeature(x)
    conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
    box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
    coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
    return conf, box, coef


class ProtoNet(nn.Module):
  def __init__(self, coef_dim):
    super().__init__()
    self.proto1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.proto2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                nn.ReLU(inplace=True))

  def forward(self, x):
    x = self.proto1(x)
    x = self.upsample(x)
    x = self.proto2(x)
    return x


class FPN(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
    self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                    nn.ReLU(inplace=True)) for _ in self.in_channels])

    self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                          nn.ReLU(inplace=True)),
                                            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                          nn.ReLU(inplace=True))])

    self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

  def forward(self, backbone_outs):
    p5_1 = self.lat_layers[2](backbone_outs[2])
    p5_upsample = self.upsample_module[1](p5_1)

    p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
    p4_upsample = self.upsample_module[0](p4_1)

    p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

    p5 = self.pred_layers[2](p5_1)
    p4 = self.pred_layers[1](p4_1)
    p3 = self.pred_layers[0](p3_1)

    p6 = self.downsample_layers[0](p5)
    p7 = self.downsample_layers[1](p6)

    return p3, p4, p5, p6, p7


class Yolact(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.coef_dim = 32

    if cfg.__class__.__name__.startswith('res101'):
      self.backbone = ResNet(layers=(3, 4, 23, 3))
      self.fpn = FPN(in_channels=(512, 1024, 2048))
    elif cfg.__class__.__name__.startswith('res50'):
      self.backbone = ResNet(layers=(3, 4, 6, 3))
      self.fpn = FPN(in_channels=(512, 1024, 2048))
    elif cfg.__class__.__name__.startswith('swin_tiny'):
      self.backbone = SwinTransformer()
      self.fpn = FPN(in_channels=(192, 384, 768))

    self.proto_net = ProtoNet(coef_dim=self.coef_dim)  # self.coef_dim = 32
    self.prediction_layers = PredictionModule(cfg, coef_dim=self.coef_dim)

    self.anchors = []
    # fpn_fm_shape = [68, 34, 17, 9, 5]
    fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
    for i, size in enumerate(fpn_fm_shape): # 遍历每一个fpn的输出的size
      # 在这里构造anchor, cfg.scales=[24,48,96,192,384], 这个应该是做了降采样的倍数
      # 这里构造的方式和faster rcnn是一样的, 就是直接在每一个特征图上铺设anchor, 不同的是, 这里只有ratio的变化(0.5,1,2)但是没有尺寸的变化
      # 也就是说没有faster rcnn的(128,256,512)的尺寸的变化。所以对于每一个特征点只是铺设了3种anchor
      # 这里的我们产生的self.anchor是一个list, 也就是1纬度的向量。因为每一个特征点对应的框的xywh都放到了一个List里面
      # 所以self.anchor的长度是74100, 它就是68*68*3*4 + 34*34*3*4 +17*17*3*4 + 9*9*3*4 + 5*5*3*4=anchor的长度是74100
      self.anchors += make_anchors(self.cfg, size, size, self.cfg.scales[i])  # len(self.anchors) = 74100, 我们在这里构造anchor

    if cfg.mode == 'train':
      self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1) # cfg.num_class-1=80

    # init weights, backbone weights will be covered later
    for name, module in self.named_modules():
      if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data)

        if module.bias is not None:
          module.bias.data.zero_()

  def load_weights(self, weight, cuda):
    if cuda:
      state_dict = torch.load(weight)
    else:
      state_dict = torch.load(weight, map_location='cpu')

    for key in list(state_dict.keys()):
      if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
        del state_dict[key]

    self.load_state_dict(state_dict, strict=True)
    print(f'Model loaded with {weight}.\n')
    print(f'Number of all parameters: {sum([p.numel() for p in self.parameters()])}\n')

  def forward(self, img, box_classes=None, masks_gt=None):
    outs = self.backbone(img)  # outs = [(1,256,136,136), (1,512,68,68), (1,1024,34,34), (1,2048,17,17)]
    outs = self.fpn(outs[1:4]) # outs = [(1,256,68,68), (1,256,34,34), (1,256,17,17), (1,256,9,9), (1,256,5,5)]
    proto_out = self.proto_net(outs[0])  # feature map P3, outs[0] = (1,256,68,68), proto_out=(1,32,136,136)
    proto_out = proto_out.permute(0, 2, 3, 1).contiguous() # (1,32,136,136)=>(1,136,136,32)

    class_pred, box_pred, coef_pred = [], [], []

    for aa in outs: # 对每一个fpn层进行遍历
      class_p, box_p, coef_p = self.prediction_layers(aa)
      class_pred.append(class_p) # class_pred = [(1,13872,81), (1,3468,81), (1,687,81), (1,243,81), (1,75,81)]
      box_pred.append(box_p) # class_pred = [(1,13872,4), (1,3468,4), (1,687,4), (1,243,4), (1,75,4)]
      coef_pred.append(coef_p) # class_pred = [(1,13872,32), (1, 3468, 32), (1,687,32), (1,243,32), (1,75,32)]

    class_pred = torch.cat(class_pred, dim=1) # (1,18525,81)
    box_pred = torch.cat(box_pred, dim=1) # (1,18525,4), 预测出来的anchor在是当前的图像下的相对坐标
    coef_pred = torch.cat(coef_pred, dim=1) # (1,18525,32) 这个算出来18525个anchor对应的一维向量(32,)的介于0~1的加权系数

    if self.training:
      seg_pred = self.semantic_seg_conv(outs[0]) # (1,80,68,68)
      return self.compute_loss(class_pred, box_pred, coef_pred, proto_out, seg_pred, box_classes, masks_gt)
    else:
      class_pred = F.softmax(class_pred, -1)
      return class_pred, box_pred, coef_pred, proto_out

  def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt):
    device = class_p.device
    class_gt = [None] * len(box_class)
    batch_size = box_p.size(0) # batch_size=1

    if isinstance(self.anchors, list):
      # self.anchors 就是把list的一维度的74100给reshaoe 成了(18525, 4)
      self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)

    num_anchors = self.anchors.shape[0] # 18525

    all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)  # (1, 18525, 4)
    conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)   # (1, 18525)
    anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)  # (1,18525,4)
    anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)  # (1, 18525)

    for i in range(batch_size):  # batch_size=1, 所以只是遍历一次
      box_gt = box_class[i][:, :-1]  # (13,4) 一共13个target
      class_gt[i] = box_class[i][:, -1].long()  # tensor([67,  0,  0,  0,  0, 67,  0,  0,  0,  0,  0,  0,  0])
      # all_offsets:(1,18525,4) 其中1代表batchsize中的第一张图像
      # conf_gt:(1,18525)这个就是18525中,元素为1的是正样本, -1的是忽略样本,也就是不参与计算
      all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                            self.anchors, class_gt[i])

    # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
    # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
    #          '0' means background, '>0' means foreground.
    # anchor_max_gt: the corresponding max IoU gt box for each anchor
    # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
    assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
           (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

    # only compute losses from positive samples
    # (1,18525), 对于18525个anchor, 看看哪些是正样本的哪些是忽略样本还有负样本。
    pos_bool = conf_gt > 0

    loss_c = self.category_loss(class_p, conf_gt, pos_bool)
    loss_b = self.box_loss(box_p, all_offsets, pos_bool)
    # pos_bool(1,18525), anchor_max_i(1,18525),
    # coef_p(1,18525,32), proto_p(1,136,136,32),
    # mask_gt(13, 544,544), 一共有13个target anchor_max_gt(1,18525,4),
    loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)
    # seg_p = (1,80,68,68), mask_gt (2,544,544), class_gt [0, 38]
    loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)
    return loss_c, loss_b, loss_m, loss_s

  def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
    # Compute max conf across batch for hard negative mining
    batch_conf = class_p.reshape(-1, self.cfg.num_classes)

    batch_conf_max = batch_conf.max()
    mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

    # Hard Negative Mining
    mark = mark.reshape(class_p.size(0), -1)
    mark[pos_bool] = 0  # filter out pos boxes
    mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

    _, idx = mark.sort(1, descending=True)
    _, idx_rank = idx.sort(1)

    num_pos = pos_bool.long().sum(1, keepdim=True)
    num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
    neg_bool = idx_rank < num_neg.expand_as(idx_rank)

    # Just in case there aren't enough negatives, don't start using positives as negatives
    neg_bool[pos_bool] = 0
    neg_bool[conf_gt < 0] = 0  # Filter out neutrals

    # Confidence Loss Including Positive and Negative Examples
    class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
    class_gt_mined = conf_gt[(pos_bool + neg_bool)]

    return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

  def box_loss(self, box_p, all_offsets, pos_bool):
    num_pos = pos_bool.sum()
    pos_box_p = box_p[pos_bool, :]
    pos_offsets = all_offsets[pos_bool, :]

    return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos

  def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
    proto_h, proto_w = proto_p.shape[1:3]  # proto_p=(1,136,136,32)
    total_pos_num = pos_bool.sum() # 统计18525的anchor中,有多少个正样本。一共有total_pos_num=30个是正样本
    loss_m = 0

    # coef_p=(1,18525,32) 其中1是batchsize, 这里相当于遍历batchsize中的每一个图像
    # 因为batch=1相当于遍历一次
    for i in range(coef_p.size(0)):
      # downsample the gt mask to the size of 'proto_p'
      # mask_gt是list of tensor, list里面的每一个元素都是当前图像待检测的mask
      # 比如当前图像待检测的mask有13张, 所mask_gt[0]=(13,544,544).unsqueeze(0)=(1,13,544,544)
      # mask_gt是从self.coco.annToMask获取的. 它里面就是一个0或者1的值。
      # proto_h=136, proto_w=136, downsampled_masks=(13,136,136), 也就是对downsampled_masks进行上采样
      downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                        align_corners=False).squeeze(0)
      downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous() # (13,136,136)=>(136,136,13)
      # binarize the gt mask because of the downsample operation
      # tensor.gt方法就是greater than的意思, 这个操作相当于downsampled_masks > 0.5, 但是这个操作没什么用
      downsampled_masks = downsampled_masks.gt(0.5).float()
      # pos_bool:正样本的anchor的id是true
      # pos_anchor_i 获取正样本的anchor对应的gt_annotation的id,因为当前的gt annotation(待检测)物体只有13个,tensor([ 8,  6,  6,  6,  6,  5,  0,  9, 11, 12, 12, 12,  7,  7, 11, 12,  7, 10, 4,  2,  4,  3,  2,  4,  4,  1,  1,  1,  1,  1])
      # pos_anchor_box 这个是anchor对应的GT的box, 因为18525中只有30个是正样本,所以pos_anchor_box是(30,4)
      # pos_coef: 是在18525个anchor中取出30个正样本对应的加权系数值, 30个样本每一个对应的是一个32通道的1维向量。所以pos_coef=(30,32)
      pos_anchor_i = anchor_max_i[i][pos_bool[i]]
      pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
      pos_coef = coef_p[i][pos_bool[i]]

      if pos_anchor_i.size(0) == 0:
        continue

      # If exceeds the number of masks for training, select a random subset
      old_num_pos = pos_coef.size(0) # 30
      if old_num_pos > self.cfg.masks_to_train:  # self.cfg.masks_to_train=100, 30 < 100, 所以不执行这个分支
        perm = torch.randperm(pos_coef.size(0))
        select = perm[:self.cfg.masks_to_train]
        pos_coef = pos_coef[select]
        pos_anchor_i = pos_anchor_i[select]
        pos_anchor_box = pos_anchor_box[select]

      num_pos = pos_coef.size(0) # 30
      # [136,136,13][:,:,tensor([8,6,6,5.....])]=>(136,136,30),取出30个正样本anchor对应的mask
      pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

      # mask assembly by linear combination
      # @ means dot product
      # Note:进行了归一化,变成了概率值, 注意这里proto_p(136,136,32)和(30,32)进行了点乘。然后进行sigmoid处理。得到的是# (136,136,30)
      mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
      # pos_anchor_box:正样本anchor对应的坐标值
      # mask_p: 把预测的mask区域给crop出来, 然后再次贴合到图里面,最终mask_p里面只是含有0和1的二值图。这一步相当于构造训练用的mask
      mask_p = crop(mask_p, pos_anchor_box)   # pos_anchor_box.shape: (30, 4)
      # TODO: grad out of gt box is 0, should it be modified?
      # TODO: need an upsample before computing loss?
      # pos_mask_gt 是30个正样本的mask,是(136,136,30)
      mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
      # mask_loss = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)

      # Normalize the mask loss to emulate roi pooling's effect on loss.
      anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
      mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

      if old_num_pos > num_pos:
        mask_loss *= old_num_pos / num_pos

      loss_m += torch.sum(mask_loss)

    return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

  def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
    # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
    batch_size, num_classes, mask_h, mask_w = segmentation_p.size() # mask_h=68, mask_w=68, numclasses=80
    loss_s = 0

    for i in range(batch_size):
      cur_segment = segmentation_p[i] # (80,68,68)
      cur_class_gt = class_gt[i] # class_gt[0]=tensor([67,  0,  0,  0,  0, 67,  0,  0,  0,  0,  0,  0,  0])
      # mask_gt[i].unsqueeze(0) = (1,13,544,544) downsampled_masks=(13,68,68)
      downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                        align_corners=False).squeeze(0)
      downsampled_masks = downsampled_masks.gt(0.5).float()

      # Construct Semantic Segmentation
      segment_gt = torch.zeros_like(cur_segment, requires_grad=False) # (80, 68, 68)
      for j in range(downsampled_masks.size(0)): # 遍历13次, 分别遍历待检测物体的预测的mask
        # segment_gt[cur_class_gt[j]]=(80, 68, 68)[0]相当于人的mask_gt和预测的mask。其中segment_gt相当于gt_annotation值,然后downsampled_masks相当于从网络输出的预测值。
        segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])
      # cur_segment是(80, 68, 68), 80个类别中的第0个类别和第38个类别被填充过, 然后和segment_gt做监督
      loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

    return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size
