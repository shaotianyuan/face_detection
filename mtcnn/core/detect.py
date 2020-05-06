import cv2
import torch
import numpy as np
from torchvision import transforms
from core.models import creat_mtcnn_net
from core.utils import generate_bounding_box, nms, convert_to_square


def detect_pnet(img, pnet_path):
    pnet, _, _ = creat_mtcnn_net(p_model_path=pnet_path)
    all_boxes = []
    net_size = 12
    current_scale = 1
    current_height, current_width, _ = img.shape

    while min(current_height, current_width) > net_size:

        image = transforms.ToTensor()(img)
        image = image.unsqueeze(0)

        cls_map, reg = pnet(image)

        cls_map = cls_map[0, 0, :, :]
        reg = reg[0, :, :]

        cls_map_np = cls_map.detach().numpy()
        reg_np = np.transpose(reg.detach().numpy(), (1, 2, 0))

        boxes = generate_bounding_box(cls_map_np, reg_np, current_scale, 0.6)

        # 缩放图片
        current_scale *= 0.709
        current_height = int(0.709 * current_height)
        current_width = int(0.709 * current_width)

        img = cv2.resize(img, (current_width, current_height), interpolation=cv2.INTER_LINEAR)

        if boxes.size == 0:
            continue

        keep = nms(boxes[:, :5], 0.5, 'Union')
        boxes = boxes[keep]

        all_boxes.append(boxes)

    if len(all_boxes) == 0:
        return None, None

    all_boxes = np.vstack(all_boxes)

    keep = nms(all_boxes[:, :5], 0.7, 'Union')
    all_boxes = all_boxes[keep]

    bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

    align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
    align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
    align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
    align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

    boxes_align = np.array([align_topx,
                            align_topy,
                            align_bottomx,
                            align_bottomy,
                            all_boxes[:, 4]])

    boxes_align = boxes_align.T

    boxes = all_boxes[:, :5]

    return boxes, boxes_align


def detect_rnet(img, dets, rnet_path):
    h, w, c = img.shape
    if dets is None:
        return None, None
    _, rnet, _ = creat_mtcnn_net(r_model_path=rnet_path)

    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])

    num_boxes = dets.shape[0]
    cropped_ims_tensors = []

    for i in range(num_boxes):
        x1, y1, x2, y2, _ = [int(c) for c in dets[i]]

        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = w - 1 if x2 > w - 1 else x2
        y2 = h - 1 if y2 > h - 1 else y2

        tmp = img[y1: y2 + 1, x1: x2 + 1, :]

        crop_im = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
        crop_im_tensor = transforms.ToTensor()(crop_im)

        cropped_ims_tensors.append(crop_im_tensor)

    feed_imgs = torch.stack(cropped_ims_tensors)
    cls_map_r, reg_r = rnet(feed_imgs)

    cls_map_np = cls_map_r.detach().numpy()
    reg_np = reg_r.detach().numpy()

    keep_inds = np.where(cls_map_np > 0.7)[0]

    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        cls = cls_map_np[keep_inds]
        reg = reg_np[keep_inds]
    else:
        return None, None

    keep = nms(boxes, 0.7)

    if len(keep) > 0:
        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
    else:
        return None, None

    bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
    bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

    align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
    align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
    align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
    align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

    boxes_align = np.vstack([align_topx,
                             align_topy,
                             align_bottomx,
                             align_bottomy,
                             keep_cls[:, 0]])

    boxes_align = boxes_align.T

    return keep_boxes, boxes_align



