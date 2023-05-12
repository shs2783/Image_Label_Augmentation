import cv2
import numpy as np
from typing import Union, List, Tuple

def crop_image(img: np.ndarray,
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

def flip_image(img: np.ndarray,
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

def rotate_image(img: np.ndarray,
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

if __name__ == '__main__':
    img = cv2.imread('./dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')
    bbox = [170, 100, 340, 350]  # xmin, ymin, xmax, ymax
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 69, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    cropped_img, cropped_bbox = crop_image(img, bbox, top=0.1, bottom=0.2, left=0.3, right=0)
    cv2.rectangle(cropped_img, (cropped_bbox[0], cropped_bbox[1]), (cropped_bbox[2], cropped_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('cropped_img', cropped_img)
    cv2.waitKey(0)

    flipped_img, flipped_bbox = flip_image(img, bbox, vertical=True, horizontal=True)
    cv2.rectangle(flipped_img, (flipped_bbox[0], flipped_bbox[1]), (flipped_bbox[2], flipped_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('flipped_img', flipped_img)
    cv2.waitKey(0)

    rotated_img, rotated_bbox = rotate_image(img, bbox, angle=45, scale=0.7)
    cv2.rectangle(rotated_img, (rotated_bbox[0], rotated_bbox[1]), (rotated_bbox[2], rotated_bbox[3]), (0, 69, 255), 2)
    cv2.imshow('rotated_img', rotated_img)
    cv2.waitKey(0)
