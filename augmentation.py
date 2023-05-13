import cv2
import numpy as np
from typing import Union, List, Tuple

def resize(img: np.ndarray,
           bbox: Union[Tuple[float], List[float], np.ndarray],
           size: Union[Tuple[int], List[int]],
           interpolation: int = cv2.INTER_LINEAR):

    img = img.copy()
    width, height = size
    img_width, img_height, _ = img.shape

    # image
    resized_img = cv2.resize(img, (width, height), interpolation=interpolation)

    # bounding box
    xmin, ymin, xmax, ymax = bbox

    width_ratio = width / img_width
    height_ratio = height / img_height

    xmin = int(xmin * width_ratio)
    xmax = int(xmax * width_ratio)
    ymin = int(ymin * height_ratio)
    ymax = int(ymax * height_ratio)

    resized_bbox = [xmin, ymin, xmax, ymax]

    return resized_img, resized_bbox

def crop(img: np.ndarray,
         bbox: Union[Tuple[float], List[float], np.ndarray],
         top: float = 0,
         bottom: float = 0,
         left: float = 0,
         right: float = 0,
         boundary: bool = True):

    assert left + right < 1, f'sum of left and right must be less than 1, but got {left + right}'
    assert top + bottom < 1, f'sum of top and bottom must be less than 1, but got {top + bottom}'

    img = img.copy()
    height, width, _ = img.shape

    # image
    top = int(top * height)
    bottom = int(bottom * height)
    left = int(left * width)
    right = int(right * width)

    _right = None if right == 0 else -right
    _bottom = None if bottom == 0 else -bottom
    cropped_img = img[top:_bottom, left:_right]

    # bounding box
    xmin, ymin, xmax, ymax = bbox
    xmin -= left
    xmax -= left
    ymin -= top
    ymax -= top

    if boundary:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, width - right - left)
        ymax = min(ymax, height - bottom - top)

    cropped_bbox = [xmin, ymin, xmax, ymax]
    return cropped_img, cropped_bbox

def center_crop(img: np.ndarray,
                bbox: Union[Tuple[float], List[float], np.ndarray],
                img_size: int,
                boundary: bool = True):

    img = img.copy()
    height, width, _ = img.shape

    # image
    center_x, center_y = width // 2, height // 2
    top = center_y - img_size // 2
    bottom = center_y + img_size // 2
    left = center_x - img_size // 2
    right = center_x + img_size // 2

    center_cropped_img = img[top:bottom, left:right]

    # bounding box
    xmin, ymin, xmax, ymax = bbox
    xmin -= left
    xmax -= left
    ymin -= top
    ymax -= top

    if boundary:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, img_size)
        ymax = min(ymax, img_size)

    center_cropped_bbox = [xmin, ymin, xmax, ymax]

    return center_cropped_img, center_cropped_bbox

def flip(img: np.ndarray,
         bbox: Union[Tuple[float], List[float], np.ndarray],
         vertical: bool = True,
         horizontal: bool = True):

    img = img.copy()
    height, width, _ = img.shape

    # image
    if vertical:
        img = cv2.flip(img, 0)
    if horizontal:
        img = cv2.flip(img, 1)

    flipped_img = img

    # bounding box
    xmin, ymin, xmax, ymax = bbox

    if vertical:
        ymin = height - ymin
        ymax = height - ymax
    if horizontal:
        xmin = width - xmin
        xmax = width - xmax

    flipped_bbox = [xmin, ymin, xmax, ymax]

    return flipped_img, flipped_bbox

def rotate(img: np.ndarray,
           bbox: Union[Tuple[float], List[float], np.ndarray],
           angle: int,
           scale: float = 1.0):

    img = img.copy()
    height, width, _ = img.shape

    # image
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (width, height))

    # bounding box  ## TODO: 구현하세욧!
    xmin, ymin, xmax, ymax = bbox
    rotated_bbox = [xmin, ymin, xmax, ymax]

    return rotated_img, rotated_bbox

def pad(img,
        bbox,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        pad_values: int = 0,
        pad_mode: str = 'constant'):

    img = img.copy()
    height, width, _ = img.shape

    # image
    padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode=pad_mode, constant_values=pad_values)

    # bounding box
    xmin, ymin, xmax, ymax = bbox
    xmin += left
    xmax += left
    ymin += top
    ymax += top
    padded_bbox = [xmin, ymin, xmax, ymax]

    return padded_img, padded_bbox

if __name__ == '__main__':
    img = cv2.imread('./dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')
    bbox = [170, 100, 340, 350]  # xmin, ymin, xmax, ymax
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 69, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    resized_img, resized_bbox = resize(img, bbox, size=(224, 224))
    cv2.rectangle(resized_img, (resized_bbox[0], resized_bbox[1]), (resized_bbox[2], resized_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('resized_img', resized_img)
    cv2.waitKey(0)

    cropped_img, cropped_bbox = crop(img, bbox, top=0.1, bottom=0.2, left=0.3, right=0)
    cv2.rectangle(cropped_img, (cropped_bbox[0], cropped_bbox[1]), (cropped_bbox[2], cropped_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('cropped_img', cropped_img)
    cv2.waitKey(0)

    center_cropped_img, center_cropped_bbox = center_crop(img, bbox, img_size=224)
    cv2.rectangle(center_cropped_img, (center_cropped_bbox[0], center_cropped_bbox[1]), (center_cropped_bbox[2], center_cropped_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('center_cropped_img', center_cropped_img)
    cv2.waitKey(0)

    flipped_img, flipped_bbox = flip(img, bbox, vertical=True, horizontal=True)
    cv2.rectangle(flipped_img, (flipped_bbox[0], flipped_bbox[1]), (flipped_bbox[2], flipped_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('flipped_img', flipped_img)
    cv2.waitKey(0)

    rotated_img, rotated_bbox = rotate(img, bbox, angle=132, scale=0.6)
    cv2.rectangle(rotated_img, (rotated_bbox[0], rotated_bbox[1]), (rotated_bbox[2], rotated_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('rotated_img', rotated_img)
    cv2.waitKey(0)

    padded_img, padded_bbox = pad(img, bbox, top=100, bottom=100, left=100, right=100, pad_values=255, pad_mode='constant')
    cv2.rectangle(padded_img, (padded_bbox[0], padded_bbox[1]), (padded_bbox[2], padded_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('padded_img', padded_img)
    cv2.waitKey(0)
