# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy_mp(x):
    x_c, y_c, w, h = x.unbind(-1)
    x0 = (x_c - 0.5 * w)
    y0 = (y_c - 0.5 * h)
    x1 = (x_c + 0.5 * w)
    y1 = (y_c + 0.5 * h)
    x1 = torch.max(x0, x1)
    y1 = torch.max(y0, y1)
    b = [x0, y0, x1, y1]
    res = torch.stack(b, dim=-1)
    # print(res.type())
    # print("cxcywh", res.shape)
    if torch.isnan(res).any():
        # print(x)
        # print(x_c, y_c, w, h )
        # print(x0, y0, x1, y1)
        # print(b)
        print(res.shape)
        print(res.type())
        print(res)
        # res[:, :2] = torch.ones(1, 2, device=torch.device('cuda')) * 0.4
        # res[:, 2:] = torch.ones(1, 2, device=torch.device('cuda')) * 0.6
        # res[:, :2] =  0.4
        # res[:, 2:] =  0.6
        # res =  res[0:1,:]
        res[:, 0] = 0.6866
        res[:, 1] = 0.5545
        res[:, 2] = 0.8551
        res[:, 3] = 0.8008
        print(res.shape)
        print(res.type())
        print(res)
        # x0 = torch.zeros(1, device=torch.device('cuda'))
        # y0 = torch.zeros(1, device=torch.device('cuda'))
        # x1 = torch.zeros(1, device=torch.device('cuda'))
        # y1 = torch.zeros(1, device=torch.device('cuda'))
        # b = [x0, y0, x1, y1]
        # res = torch.stack(b, dim=-1)

    return res


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# added by mp
def box_xywh_to_xyxy(x):
    x0, y0, w, h = x.unbind(-1)
    x1 = (x0 + w)
    y1 = (y0 + h)
    # print("pre", x0[0], y0[0], x1[0], y1[0])

    x1 = torch.max(x0, x1)
    y1 = torch.max(y0, y1)

    # print("after", x0[0], y0[0], x1[0], y1[0])

    b = [x0, y0, x1, y1]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # print("generalized_box_iou")
    # print(boxes1.shape)
    # print(boxes1[0])

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
