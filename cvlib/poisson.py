"""
Edited from 
https://github.com/willemmanuel/poisson-image-editing
"""

import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
import cv2
import time
import numba as nb

 # Helper enum
OMEGA = 0
DEL_OMEGA = 1
OUTSIDE = 2
CHANNELS = ["Blue","Green","Red"]

class Poisson():
   
    def __init__(self, target, source, mask):
        self.target_bgr = target.copy()
        self.source_bgr = source.copy()
        self.mask = mask.copy()
        self.HEIGHT, self.WIDTH = mask.shape[0], mask.shape[1]

    # Determine if a given index is inside omega, on the boundary (del omega),
    # or outside the omega region
    def point_location(self,index):
        if self.in_omega(index) == False:
            return OUTSIDE
        if self.edge(index) == True:
            return DEL_OMEGA
        return OMEGA

    # Determine if a given index is either outside or inside omega
    def in_omega(self,index):
        return self.mask[index] == 1

    # Deterimine if a given index is on del omega (boundary)
    def edge(self,index):
        if self.in_omega(index) == False: return False
        for pt in self.get_surrounding(index):
            # If the point is inside omega, and a surrounding point is not,
            # then we must be on an edge
            if self.in_omega(pt) == False: return True
        return False

    # Apply the Laplacian operator at a given index
    def lapl_at_index(self, index):
        i,j = index
        val = (4 * self.source[i,j])    \
            - (1 * (self.source[i+1,j] if (i < self.HEIGHT-1) else 0) ) \
            - (1 * (self.source[i-1, j] if (i > 0) else 0) ) \
            - (1 * (self.source[i, j+1] if (j < self.WIDTH-1) else 0) ) \
            - (1 * (self.source[i, j-1] if (j > 0) else 0 ) )
        return val

    # Find the indicies of omega, or where the mask is 1
    def mask_indicies(self):
        nonzero = np.nonzero(self.mask)
        return list(zip(nonzero[0], nonzero[1]))

    # Get indicies above, below, to the left and right
    def get_surrounding(self,index):
        i,j = index
        surroundings = []
        if i < self.HEIGHT-1:
            surroundings.append((i+1,j))
        if i > 0:
            surroundings.append((i-1,j))
        if j < self.WIDTH-1:
            surroundings.append((i,j+1))
        if j > 0:
            surroundings.append((i,j-1))
        return surroundings
        #return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

    # Create the A sparse matrix
    def poisson_sparse_matrix(self):
        # N = number of points in mask
        A = lil_matrix((self.N,self.N))
        # Set up row for each point in mask
        for i,index in enumerate(self.indicies):
            # Should have 4's diagonal
            A[i,i] = 4
            # Get all surrounding self.indicies
            for x in self.get_surrounding(index):
                # If a surrounding point is in the mask, add -1 to index's
                # row at correct position
                if x not in self.indicies: continue
                j = self.indicies.index(x)
                A[i,j] = -1
        return A

    def poisson_b_matrix(self):
        b = np.zeros(self.N)
        for i,index in enumerate(self.indicies):
            b[i] = self.lapl_at_index(index)
            # If on boundry, add in target intensity
            # Creates constraint lapl source = target at boundary
            if self.point_location(index) == DEL_OMEGA:
                for pt in self.get_surrounding(index):
                    if self.in_omega(pt) == False:
                        b[i] += self.target[pt]
        return b

    def solve_poisson(self):
        self.indicies = self.mask_indicies()
        self.N = len(self.indicies)
        # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
        A = self.poisson_sparse_matrix()
        #Create poisson b matrix
        b = self.poisson_b_matrix()
        # Solve for x, unknown intensities
        x = linalg.cg(A, b)
        # Copy target photo, make sure as int
        composite = np.copy(self.target).astype(int)
        # Place new intensity on target at given index 
        for i,index in enumerate(self.indicies):
            composite[index] = x[0][i]
        return composite

    #Entering function to poisson
    def process(self):
        blend_stack = []
        print("Poisson blending started")
        #Solve Poisson Eq. for each channel
        for i in range(3):
            self.target = self.target_bgr[:,:,i]
            self.source = self.source_bgr[:,:,i]
            start = time.time()
            blend_stack.append(self.solve_poisson())
            print(f"{CHANNELS[i]} channel finished in {time.time() - start} seconds.")
        blended = cv2.merge(blend_stack)
        return blended

    # Naive blend, puts the source region directly on the target.
    # Useful for testing
    def preview(source, target, mask):
        return (target * (1.0 - mask)) + (source * (mask))
