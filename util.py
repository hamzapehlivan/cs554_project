import numpy as np

def patch_from_corners(image, upperLeft, lowerRight):  
    return image[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1]
def update_image(image, patch, upperLeft, lowerRight):
    image[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1] = patch

def checkInputs(target, reference, mask):
    if target is None:
        raise "Target is None"
    if reference is None:
        raise "Reference is None"
    if mask is None:
        raise "Mask is None"