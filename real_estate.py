import argparse
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import util
from cvlib.aligner import Aligner
from cvlib.examplar_inpainter import ExamplarInpainter
from cvlib.poisson import Poisson

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, default="data/TransFill_Testing_Data/Small_Set/target", help='Images that will have holes')
parser.add_argument('--reference_dir', type=str, default="data/TransFill_Testing_Data/Small_Set/source", help='Reference image')
parser.add_argument('--hole_dir', type=str, default="data/TransFill_Testing_Data/Small_Set/hole", help="Directory to input holes.")
parser.add_argument('--gui', action="store_true", help='Let user refine inpainting result.')
parser.add_argument('--eval', action="store_true", help="Whether to run evaluation code.")
parser.add_argument('--debug', action="store_true", help="Whether to save intermediate results for debugging")
parser.add_argument('--save_dir', type=str, default='result', help='Where to save output images')
args = parser.parse_args()
gui = args.gui
eval = args.eval
target_dir = args.target_dir
reference_dir = args.reference_dir
hole_dir = args.hole_dir
save_dir = args.save_dir
debug = False
os.makedirs(save_dir, exist_ok=True)

#Mouse Handler Function in GUI.
upperLeft = None
lowerRight = None
image = None
region_list = []
def get_regions(event, x, y, flags, param):

    global upperLeft, lowerRight, region_list, image
    if event == cv2.EVENT_LBUTTONDOWN:
        upperLeft = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        lowerRight = (x,y)
        region_list.append((upperLeft, lowerRight))
        # draw a rectangle around the region of interest
        cv2.rectangle(image, upperLeft, lowerRight, (0, 255, 0), 2)
        cv2.imshow("inpainter", image)

#Read Dataset
target_images = sorted(os.listdir(target_dir))
reference_images = sorted(os.listdir(reference_dir))
hole_images = sorted(os.listdir(hole_dir))
dataset_size = len(target_images)
print("Dataset size is: ", dataset_size)
#If there is one hole, apply it to all images.
if len(hole_images) == 1:
    hole_images = hole_images * dataset_size
ssim_list = []
psnr_list = []
for i in range(dataset_size):
    image_target = cv2.imread(os.path.join(target_dir, target_images[i]), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(hole_dir, hole_images[i]), cv2.IMREAD_GRAYSCALE) / 255
    mask = np.expand_dims(1-mask, -1).astype(np.uint8)
    masked_target = image_target * mask
    image_ref = cv2.imread(os.path.join(reference_dir, reference_images[i]), cv2.IMREAD_COLOR)
    util.checkInputs(image_target, image_ref, mask)

    if debug:
        cv2.imwrite("results/ref.png", image_ref)
        cv2.imwrite("results/target.png", image_target)
        cv2.imwrite("results/masked_image.png", masked_target)

    aligner = Aligner(masked_target, image_ref, mask, debug=debug)
    aligned, warped = aligner.align()

    ## Correct unaligned regions with Examplar Based Inpainting
    if gui:
        image = aligned.copy()
        clone = image.copy()
        cv2.namedWindow("inpainter")
        cv2.setMouseCallback("inpainter", get_regions)
        while True:
            cv2.imshow("inpainter", image)
            key = cv2.waitKey(1)
            # Redraw rectangles
            if key == ord("r"):
                image = clone.copy()
                region_list = []
            # Continue normal execution
            elif key == ord("c"):
                break
        cv2.destroyAllWindows()
        for i in range(len(region_list)):
            upperLeft = region_list[i][0]
            lowerRight = region_list[i][1]
            ## Inputs
            input_image = util.patch_from_corners(aligned, region_list[i][0], region_list[i][1])
            zero_regions = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) == 0 #Find the zero region
            mask_crop = util.patch_from_corners(1-mask, region_list[i][0], region_list[i][1])[:,:,0]
            input_mask = zero_regions & mask_crop
            #Dilate mask by 1 to increase robustness
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            input_mask = cv2.dilate(input_mask, kernel, iterations=1)
            #Examplar based inpainting
            examplar_inpainter = ExamplarInpainter(input_image, input_mask)
            result = examplar_inpainter.inpaint()
            util.update_image(warped, result, upperLeft, lowerRight)
            if debug:
                cv2.imwrite("results/examplar.png", warped)

    poisson = Poisson( image_target,warped, 1-mask)
    blended = poisson.process()
    blended = blended.astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'result{i}.png'), blended)
    if eval:
        psnr = PSNR(image_target, blended)
        ssim = SSIM(image_target, blended, channel_axis=2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f"Image {i}, PSNR: {psnr}, SSIM: {ssim}")
print("SSIM score: ", np.mean(ssim_list))
print("PSNR score: ", np.mean(psnr_list))


