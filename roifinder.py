# roifinder.py

import matplotlib.patches as patches
from skimage import io, color, exposure, morphology, measure, transform
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
def roi_detector(image, debug=False):
    """
    对输入图像进行HSV阈值、形态学处理和连通区域分析，
    返回满足一定面积和形状要求的所有候选RoI（边界框列表）以及中间结果。
    如果 debug 为 True，则同时返回每个区域是否入选及其拒绝原因。
    """
    # 只取前三通道，确保是RGB
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    # 1) 转到HSV空间（不使用红色增强）
    hsv = color.rgb2hsv(image)


    hue_mask = (hsv[:, :, 0] > 0.95) | (hsv[:, :, 0] < 0.09)
    sat_mask = hsv[:, :, 1] > 0.4
    val_mask = hsv[:, :, 2] > 0.15

    binary_mask = hue_mask & sat_mask & val_mask

    # 3) 形态学操作
    selem = morphology.disk(1)
    opened_mask = morphology.binary_opening(binary_mask, selem)
    refined_mask = morphology.binary_closing(opened_mask, selem)

    # 4) 连通区域标记
    labeled_mask, num_features = label(refined_mask)
    slices = find_objects(labeled_mask)

    # 5) 根据面积和宽高比过滤，收集RoI边界框，同时记录调试信息
    rois = []
    rejection_info = []  # 用于记录每个区域的判断信息
    for i, slc in enumerate(slices):
        if slc is None:
            continue
        # 当前区域在 labeled_mask 中的标签为 i+1
        region_mask = (labeled_mask[slc] == (i + 1))
        area = np.sum(region_mask)
        info_str = f"Region {i}: area = {area}. "
        passed = True

        # 面积判断
        if area <= 280:
            info_str += "Rejected: area too small (<=280)."
            passed = False
        else:
            minr, maxr = slc[0].start, slc[0].stop
            minc, maxc = slc[1].start, slc[1].stop
            height = maxr - minr
            width = maxc - minc
            ratio = width / height
            info_str += f"Width/Height = {ratio:.2f}. "
            # 宽高比判断（要求区域接近正方形或圆形）
            if ratio >= 1.8:
                info_str += "Rejected: width/height ratio too high (>=1.8)."
                passed = False

        if passed:
            # 收录该区域
            minr, maxr = slc[0].start, slc[0].stop
            minc, maxc = slc[1].start, slc[1].stop
            rois.append((minr, minc, maxr, maxc))
            info_str += "Accepted."
        rejection_info.append(info_str)

    if debug:
        return rois, binary_mask, opened_mask, refined_mask, hue_mask, sat_mask, val_mask, rejection_info
    else:
        return rois, binary_mask, opened_mask, refined_mask, hue_mask, sat_mask, val_mask
def visualize_refined_mask_with_rois(refined_mask, rois):
    """
    在 refined mask 上绘制候选区域（怀疑区域）的边界框，
    用于可视化调试。
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(refined_mask, cmap='gray')
    ax = plt.gca()
    for roi in rois:
        minr, minc, maxr, maxc = roi
        rect = patches.Rectangle((minc, minr), maxr-minc, maxr-minr,
                                 edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(rect)
    plt.title("Refined Mask with Suspect Regions")
    plt.axis('off')
    plt.show()

