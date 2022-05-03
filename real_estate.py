import cv2
import numpy as np
from cvlib.confidence_finder import ConfidenceFinder
from cvlib.aligner import Aligner

def checkInputs(target, reference, mask, debug=True):
    if target is None:
        raise "Target is None"
    if reference is None:
        raise "Reference is None"
    if mask is None:
        raise "Mask is None"

#data/TransFill_Testing_Data/Real_Set/target/3_target.png
#data/TransFill_Testing_Data/Small_Set/target/21d5b2ebb33325f9_10_target_x1_GT.png
image_target = cv2.imread("data/TransFill_Testing_Data/Real_Set/target/1_target.png", cv2.IMREAD_COLOR)
#data/TransFill_Testing_Data/Real_Set/hole/3_hole.png
#data/TransFill_Testing_Data/Small_Set/hole/hole.png
mask = cv2.imread("data/TransFill_Testing_Data/Real_Set/hole/1_hole.png", cv2.IMREAD_GRAYSCALE) /255
mask = np.expand_dims(1-mask, -1).astype(np.uint8)
masked_target = image_target * mask
#data/TransFill_Testing_Data/Real_Set/source/3_source.png
#data/TransFill_Testing_Data/Small_Set/source/21d5b2ebb33325f9_10_target_x1_REFO.png
image_ref = cv2.imread("data/TransFill_Testing_Data/Real_Set/source/1_source.png", cv2.IMREAD_COLOR)
debug = True

checkInputs(image_target, image_ref, mask, debug=debug)

if debug:
    cv2.imwrite("results/ref.png", image_ref)
    cv2.imwrite("results/target.png", image_target)
    cv2.imwrite("results/masked_image.png", masked_target)

aligner = Aligner(masked_target, image_ref, mask, debug=debug)
warped = aligner.align()
#finder = ConfidenceFinder()

#Find Mask Corners



