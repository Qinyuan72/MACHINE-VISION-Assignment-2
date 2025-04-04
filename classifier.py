# classifier.py

import numpy as np
from skimage import color, transform, exposure

def load_exemplars(filepath):
    """
    加载1-NN描述符向量文件，每行第一个元素为类别（例如40, 50, 60, ...，-1表示非标志），
    后续元素为描述符向量。
    """
    data = np.load(filepath)
    categories = data[:, 0]
    descriptors = data[:, 1:]
    return categories, descriptors

def roi_to_vector(roi):
    """
    将ROI图像转换为64x64灰度图，对比度增强后展平，并归一化为单位向量。
    """
    if roi.ndim == 3 and roi.shape[-1] == 4:
        roi = roi[..., :3]
    gray = color.rgb2gray(roi)
    resized = transform.resize(gray, (64, 64), anti_aliasing=True)
    enhanced = exposure.equalize_adapthist(resized)
    vector = enhanced.flatten()
    vector = vector - np.mean(vector)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector /= norm
    return vector

def classify_roi(vector, categories, descriptors, distance_threshold=1.2, debug=False):
    """
    使用1-NN分类器：计算向量与所有描述符之间的欧氏距离，找到最近邻。
    如果最近邻类别为 -1，或者最近邻距离超过设定阈值，则返回 -1（表示非标志）。
    当 debug 为 True 时，同时返回调试信息。
    """
    distances = np.linalg.norm(descriptors - vector, axis=1)
    nearest_index = np.argmin(distances)
    min_dist = distances[nearest_index]
    predicted_label = categories[nearest_index]
    debug_info = f"Min distance: {min_dist:.3f}. "

    # 判断是否匹配到负样本
    if predicted_label == -1:
        debug_info += "Matched negative sample (-1)."
        return (-1, debug_info) if debug else -1

    # 判断距离是否超过阈值
    if min_dist > distance_threshold:
        debug_info += f"Distance {min_dist:.3f} exceeds threshold {distance_threshold}."
        return (-1, debug_info) if debug else -1

    debug_info += "Accepted."
    return (predicted_label, debug_info) if debug else predicted_label

