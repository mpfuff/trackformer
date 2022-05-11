#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from math import sqrt
from os import path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from cycler import cycler as cy
from matplotlib import colors
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

matplotlib.use('Agg')


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # mpmpmp
    nlabels = max(1, nlabels)

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                              boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def objective_5(x, a, b, c, d, e, f):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f


def objective_3(x, a, b, c, d):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + d


def objective_2(x, a, b, c):
    return (a * x) + (b * x ** 1.5) + c


def objective_2a(x, a, b, c, d, e):
    return (a * (x + d)) + (b * (x + e) ** 2) + c


def objective_1(x, a, b):
    return (a * x) + b


def objective_exp(x, a, b, c):
    return a + b * np.exp(c * x)


def plot_sequence(tracks, data_loader, output_dir, write_images, generate_attention_maps, plot_traces=False):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    # cyl = cy('ec', COLORS)
    # loop_cy_iter = cyl()
    # styles = defaultdict(lambda: next(loop_cy_iter))

    # cmap = plt.cm.get_cmap('hsv', )
    cmap = rand_cmap(len(tracks), type='bright', first_color_black=False, last_color_black=False)

    # if generate_attention_maps:
    #     attention_maps_per_track = {
    #         track_id: (np.concatenate([t['attention_map'] for t in track.values()])
    #                    if len(track) > 1
    #                    else list(track.values())[0]['attention_map'])
    #         for track_id, track in tracks.items()}
    #     attention_map_thresholds = {
    #         track_id: np.histogram(maps, bins=2)[1][1]
    #         for track_id, maps in attention_maps_per_track.items()}

    # _, attention_maps_bin_edges = np.histogram(all_attention_maps, bins=2)

    traces = {}

    for frame_id, frame_data in enumerate(tqdm.tqdm(data_loader)):
        img_path = frame_data['img_path'][0]
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]
        height, width, _ = img.shape

        fig = plt.figure()
        fig.set_size_inches(width / 96, height / 96)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)

        if generate_attention_maps:
            attention_map_img = np.zeros((height, width, 4))

        do_fit = True
        degree_of_polynom = 5
        # last_to_fit > Degree of Polynom
        last_to_fit = 40
        # min_frozen_length = 80

        diagonale = sqrt(height ** 2 + width ** 2)
        dist_fract = diagonale * 0.1

        for track_id, track_data in tracks.items():
            # if len(list(track_data.keys())) < min_frozen_length:
            #     continue
            if frame_id in track_data.keys():
                bbox = track_data[frame_id]['bbox']

                if 'mask' in track_data[frame_id]:
                    mask = track_data[frame_id]['mask']
                    mask = np.ma.masked_where(mask == 0.0, mask)

                    ax.imshow(mask, alpha=0.5, cmap=colors.ListedColormap([cmap(track_id)]))

                    annotate_color = 'white'
                else:
                    if plot_traces:
                        if track_id not in traces:
                            traces[track_id] = {'trace': [], 'cmap': None}
                        # center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        # traces[track_id].append(center)

                        bottom = ((bbox[0] + bbox[2]) / 2, bbox[3])
                        traces[track_id]['trace'].append(bottom)

                        if len(traces[track_id]['trace']) > degree_of_polynom:
                            # https://machinelearningmastery.com/curve-fitting-with-python/
                            plot_trace(ax, cmap, do_fit, traces, track_id, dist_fract, degree_of_polynom)

                        if plot_traces == 'freeze':
                            for tr_id, value in traces.items():
                                if tr_id == track_id:
                                    continue
                                np_line = np.asarray(value['trace'])
                                # if len(np_line) <= min_frozen_length:
                                #     continue
                                if get_total_trace_length(np_line) < dist_fract:
                                    continue
                                x = np_line[:, 0]
                                y = np_line[:, 1]
                                ax.add_line(plt.Line2D(x, y, linewidth=2, color=value['cmap']))

                    ax.add_patch(
                        plt.Rectangle(
                            (bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            fill=False,
                            linewidth=2.0,
                            color=cmap(track_id),
                        ))

                    annotate_color = cmap(track_id)

                if write_images == 'debug':
                    ax.annotate(
                        f"{track_id} ({track_data[frame_id]['score']:.2f})",
                        (bbox[0] + (bbox[2] - bbox[0]) / 2.0, bbox[1] + (bbox[3] - bbox[1]) / 2.0),
                        color=annotate_color, weight='bold', fontsize=12, ha='center', va='center')

                if 'attention_map' in track_data[frame_id]:
                    attention_map = track_data[frame_id]['attention_map']
                    attention_map = cv2.resize(attention_map, (width, height))

                    # attention_map_img = np.ones((height, width, 4)) * cmap(track_id)
                    # # max value will be at 0.75 transparency
                    # attention_map_img[:, :, 3] = attention_map * 0.75 / attention_map.max()

                    # _, bin_edges = np.histogram(attention_map, bins=2)
                    # attention_map_img[:, :][attention_map < bin_edges[1]] = 0.0

                    # attention_map_img += attention_map_img

                    # _, bin_edges = np.histogram(attention_map, bins=2)

                    norm_attention_map = attention_map / attention_map.max()

                    high_att_mask = norm_attention_map > 0.25  # bin_edges[1]
                    attention_map_img[:, :][high_att_mask] = cmap(track_id)
                    attention_map_img[:, :, 3][high_att_mask] = norm_attention_map[high_att_mask] * 0.5

                    # attention_map_img[:, :] += (np.tile(attention_map[..., np.newaxis], (1,1,4)) / attention_map.max()) * cmap(track_id)
                    # attention_map_img[:, :, 3] = 0.75

        if generate_attention_maps:
            ax.imshow(attention_map_img, vmin=0.0, vmax=1.0)

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(osp.join(output_dir, osp.basename(img_path)), dpi=96)
        plt.close()


def get_total_trace_length(np_line: np.array):
    x0 = np_line[0][0]
    y0 = np_line[0][1]
    x1 = np_line[-1][0]
    y1 = np_line[-1][1]
    length = sqrt((x0-x1)**2 + (y0-y1)**2)
    return length

def get_trace_length(np_line: np.array, dist_fract):
    length = 0
    num_sections = 0
    x0, y0 = 0, 0
    for i, (x, y) in enumerate(np_line[::-1]):
        if i == 0:
            x0, y0 = x, y
        if i > 0:
            length = sqrt((x - x0) ** 2 + (y - y0) ** 2)
            num_sections += 1
        if length > dist_fract:
            break
    return length, num_sections


def plot_trace(ax, cmap, do_fit: bool, traces, track_id, dist_fract, degree_of_polynom):
    use_moving_average = True
    trace = traces[track_id]['trace']
    np_line = np.asarray(trace)
    x = np_line[:, 0]
    y = np_line[:, 1]
    length, num_sections = get_trace_length(np_line, dist_fract)
    min_to_fit = max(degree_of_polynom, num_sections)
    to_fit = min(len(x), min_to_fit)
    last_x = x[-to_fit:]
    last_y = y[-to_fit:]
    if do_fit:
        if use_moving_average:
            xs = np.arange(to_fit)
            popt_x, _ = curve_fit(objective_1, xs, last_x)
            a, b = popt_x
            # a, b, c, d = popt_x
            last_x = objective_1(xs, a, b)
            # last_x = objective_3(xs, a, b, c, d)

            popt_y, _ = curve_fit(objective_1, xs, last_y)
            a, b = popt_y
            # a, b, c, d = popt_y
            last_y = objective_1(xs, a, b)
            # last_y = objective_3(xs, a, b, c, d)

            # sections = len(last_x)
            # x_average = np.sum(last_x) / sections
            # y_average = np.sum(last_y) / sections
            # last_x[:] = x_average
            # last_y[:] = y_average
            x[-to_fit:] = last_x
        else:
            popt, _ = curve_fit(objective_3, last_x, last_y)
            # a, b, c, d, e, f = popt
            # a, b = popt
            # a, b, c = popt
            a, b, c, d = popt
            # a, b, c, d, e = popt
            # last_y = objective_1(last_x, a, b)
            last_y = objective_3(last_x, a, b, c, d)
            # last_y = objective_2a(last_x, a, b, c, d, e)
        y[-to_fit:] = last_y

    traces[track_id]['trace'] = trace[:-to_fit]
    for lx, ly in zip(last_x, last_y):
        traces[track_id]['trace'].append((lx, ly))

    traces[track_id]['cmap'] = cmap(track_id)
    ax.add_line(plt.Line2D(x, y, linewidth=2, color=cmap(track_id)))


def interpolate_tracks(tracks):
    for i, track in tracks.items():
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, data in track.items():
            frames.append(f)
            x0.append(data['bbox'][0])
            y0.append(data['bbox'][1])
            x1.append(data['bbox'][2])
            y1.append(data['bbox'][3])

        if frames:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bbox = np.array([
                    x0_inter(f),
                    y0_inter(f),
                    x1_inter(f),
                    y1_inter(f)])
                tracks[i][f]['bbox'] = bbox
        else:
            tracks[i][frames[0]]['bbox'] = np.array([
                x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                  pred_ctr_y - 0.5 * pred_h,
                                  pred_ctr_x + 0.5 * pred_w,
                                  pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, im_shape[1] - 1),
        boxes[:, :, 1].clamp(0, im_shape[0] - 1),
        boxes[:, :, 2].clamp(0, im_shape[1] - 1),
        boxes[:, :, 3].clamp(0, im_shape[0] - 1)
    ], 2).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


def get_mot_accum(results, seq_loader):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for frame_id, frame_data in enumerate(seq_loader):
        gt = frame_data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, gt_box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(gt_box[0])

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (gt_boxes[:, 0],
                 gt_boxes[:, 1],
                 gt_boxes[:, 2] - gt_boxes[:, 0],
                 gt_boxes[:, 3] - gt_boxes[:, 1]), axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, track_data in results.items():
            if frame_id in track_data:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(track_data[frame_id]['bbox'])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (track_boxes[:, 0],
                 track_boxes[:, 1],
                 track_boxes[:, 2] - track_boxes[:, 0],
                 track_boxes[:, 3] - track_boxes[:, 1]), axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=True):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall, )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names, )
    return summary, str_summary
