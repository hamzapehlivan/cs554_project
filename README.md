# CS554 Project - Reference Based Image Inpainting

Using a reference image, this algorithm fills the missing areas in the target image using image alignment, exemplar based inpainting and Poisson blending.  
Dataset can be obtained from TransFill repository.

Environment file is given so that users can replicate the results.

Usage:
`python real_estate.py --eval --gui --target_dir=(path_to_target) --reference_dir=(path_to_reference) --hole_dir=(path_to_hole) `  
--eval to measure SSIM and PSNR scores.   
--gui to select a region for exemplar based inpainting.  

The rest is should be the paths to target, reference and the hole.

Exemplar based inpainting code borrows [from](github.com/NazminJuli/Criminisi-Inpainting)   
Poisson editing code borrows [from](github.com/willemmanuel/poisson-image-editing)

cvlib/aligner.py -> Image Alignment code  
cvlib/examplar_inpainter.py -> Exemplar based inpainting, contextualized priority is implemented in computeTarget function.  
cvlib/poisson.py -> Poisson blending  

We explained the parts we implemented in the paper.   

We also worked with face images, however, they were quite simple to inpaint,so , we move to RealEstate10k dataset. That is why there is a file called face_detector.py in cvlib folder. 
