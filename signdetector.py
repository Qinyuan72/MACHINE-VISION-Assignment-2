# signdetector.py

import matplotlib.patches as patches
from classifier import load_exemplars, roi_to_vector, classify_roi
from skimage import io
from roifinder import roi_detector, visualize_refined_mask_with_rois

def sign_detector(image_path, exemplar_path):
    image = io.imread(image_path)

    rois, binary_mask, opened_mask, refined_mask, hue_mask, sat_mask, val_mask, rejection_info = roi_detector(image,debug=True)
    # visualize_refined_mask_with_rois(refined_mask, rois)

    for info in rejection_info:
        print(info)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(hue_mask, cmap='gray')
    ax[1].set_title('Hue Mask')
    ax[1].axis('off')

    ax[2].imshow(sat_mask, cmap='gray')
    ax[2].set_title('Saturation Mask')
    ax[2].axis('off')

    ax[3].imshow(val_mask, cmap='gray')
    ax[3].set_title('Value Mask')
    ax[3].axis('off')

    ax[4].imshow(binary_mask, cmap='gray')
    ax[4].set_title('Binary Mask')
    ax[4].axis('off')

    ax[5].imshow(refined_mask, cmap='gray')
    ax[5].set_title('Refined Mask')
    ax[5].axis('off')

    plt.tight_layout()
    plt.show()

    categories, descriptors = load_exemplars(exemplar_path)

    detected_speeds = []  # 用于存放检测到的速度（数值）

    for roi in rois:
        roi_img = image[roi[0]:roi[2], roi[1]:roi[3]]
        vector = roi_to_vector(roi_img)
        label, debug_msg = classify_roi(vector, categories, descriptors, distance_threshold=1.2, debug=True)
        print(f"ROI at {roi} classified as {label}. {debug_msg}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    for (minr, minc, maxr, maxc) in rois:
        roi_img = image[minr:maxr, minc:maxc]
        roi_vec = roi_to_vector(roi_img)
        label = classify_roi(roi_vec, categories, descriptors)

        # 如果分类结果为 -1，说明此ROI不是有效的速度标志，跳过绘制或另做标记
        if label == -1:
            continue

        if label != -1:
            detected_speeds.append(label)

        rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(minc, minr, f"Speed: {label}", color='white',
                fontsize=12, verticalalignment='top')

    ax.set_title('Detected Regions with 1-NN Classification')
    ax.axis('off')
    plt.show()

    return detected_speeds


import os
import time
import matplotlib.pyplot as plt

def group_test_images(folder_path, exemplar_path):
    """
    文件分组
    """
    groups = {
        1: [
            "40-0001x1.png", "40-0002x1.png",
        ],
        2: [
            "50-0001x1.png", "50-0002x1.png",
            "50-0003x1.png", "50-0004x2.png",
            "50-0005x1.png"
        ],
        3: [
            "60-0001x1.png", "60-0002x1.png",
            "60-0003x1.png", "60-0004x2.png",
            "60-0005x1.png"
        ],
        4: [
            "80-0001x1.png", "80-0002x2.png",
            "80-0003x2.png", "80-0004x2.png",
            "80-0005x1.png"

        ],
        5: [
            "100-0001x1.png", "100-0002x1.png",
            "100-0003x1.png", "100-0004x2.png",
            "100-0005x1.png"
        ],
        6: [
            "120-0001x1.png", "120-0002x2.png",
            "120-0003x1.png", "120-0004x1.png",
            "120-0005x1.png"
        ],
        7:[
            "50-0002x1.png","80-0003x2.png"
        ]
    }

    group_number = input(f"Which group do you want to test?1-40 2-50 3-60 4-80 5-100 6-120 Available: {list(groups.keys())} ")
    try:
        group_number = int(group_number)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

    if group_number not in groups:
        print(f"Group {group_number} not defined. Available groups are: {list(groups.keys())}")
        return

    images_to_test = groups[group_number]
    print(f"\nNow testing group {group_number} with images: {images_to_test}")

    for img in images_to_test:
        full_path = os.path.join(folder_path, img)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue

        print(f"\nProcessing {img} ...")
        detected_speeds = sign_detector(full_path, exemplar_path)
        count = len(detected_speeds)
        if count == 0:
            print("No speed signs detected.")
        elif count == 1:
            print(f"detected 1 speed sign: {detected_speeds[0]}km/h")
        else:
            speeds_str = ", ".join([f"{s}km/h" for s in detected_speeds])
            print(f"detected {count} speed signs: {speeds_str}")

        plt.close('all')
        time.sleep(1)

if __name__ == "__main__":
    folder_path = "image"
    exemplar_path = "1-NN-descriptor-vects.npy"
    group_test_images(folder_path, exemplar_path)

