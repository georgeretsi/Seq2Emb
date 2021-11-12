import numpy as np

from skimage.transform import resize
import cv2

def affine_transformation(img, m=1.0, s=.2, border_value=None):
    h, w = img.shape[0], img.shape[1]
    src_point = np.float32([[w / 2.0, h / 3.0],
                            [2 * w / 3.0, 2 * h / 3.0],
                            [w / 3.0, 2 * h / 3.0]])
    random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if border_value is None:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
    return warped_img

def image_resize(img, height=None, width=None):

    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)
    return word_img


def average_precision(ret_vec_relevance, gt_relevance_num=None):
    '''
    Computes the average precision from a list of relevance items

    Params:
        ret_vec_relevance: A 1-D numpy array containing ground truth (gt)
            relevance values
        gt_relevance_num: Number of relevant items in the data set
            (with respect to the ground truth)
            If None, the average precision is calculated wrt the number of
            relevant items in the retrieval list (ret_vec_relevance)

    Returns:
        The average precision for the given relevance vector.
    '''
    if ret_vec_relevance.ndim != 1:
        raise ValueError('Invalid ret_vec_relevance shape')

    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap