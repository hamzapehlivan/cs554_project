import numpy as np
import cv2
import math

class ConfidenceFinder():
    #Mask is 1 for masked areas.
    def __init__(self, inputImage, mask, reference, halfPatchWidth = 12):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask[:,:,0])
        self.reference = reference.copy()
        self.updatedMask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.result = np.ndarray(shape = inputImage.shape, dtype = inputImage.dtype)
        self.halfPatchWidth = halfPatchWidth #Patch size= halfPatchWidth*2+1 x halfPatchWidth*2+1 (9x9 default)
        self.process()
    
    def process(self):
        self.initializeMats()
        self.calculateGradients()
        self.computeFillFront()
        self.computeConfidence()
        self.computeData()
        self.computePriority()
        #x,y = self.fillFront[self.prior]
        x,y = 103,9
        upperLeft, lowerRight = self.getPatch((x,y))
        out = cv2.rectangle(self.inputImage.copy(), upperLeft, lowerRight, (0,0,255), thickness=1)
        cv2.imwrite("results/rectangle.png", out)
        out = cv2.rectangle(self.reference.copy(), upperLeft, lowerRight, (0,0,255), thickness=1)
        cv2.imwrite("results/rectangle2.png", out)
        patch_target = self.inputImage[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1]  
        patch_source= self.reference[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1]  
        patch_mask = self.mask[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1]
        patch_mask = np.expand_dims(patch_mask, -1)
        patch_mixed = patch_source*patch_mask + patch_target*(1-patch_mask)

        # patch_target = cv2.cvtColor(patch_target, cv2.COLOR_BGR2HSV)
        # patch_source = cv2.cvtColor(patch_source, cv2.COLOR_BGR2HSV)
        # patch_source[:,:,-1] = patch_source[:,:,-1] -3
        # patch_source = cv2.cvtColor(patch_source, cv2.COLOR_HSV2BGR)
        out = self.inputImage.copy()
        out[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1] = patch_target
        cv2.imwrite("results/copied.png", out)

        blended = cv2.seamlessClone(patch_source, patch_target, np.ones_like(patch_source), 
                    (self.halfPatchWidth, self.halfPatchWidth),cv2.MIXED_CLONE)
        out = self.inputImage.copy()
        out[upperLeft[1]:lowerRight[1]+1, upperLeft[0]:lowerRight[0]+1] = blended
        cv2.imwrite("results/blended.png", out)
        print()
    def initializeMats(self):
        # _, self.confidence = cv2.threshold(self.mask, 0.5, 255, cv2.THRESH_BINARY)
        # _, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        self.confidence = 1 - self.mask.copy()
        
        self.sourceRegion = np.copy(self.confidence)
        self.sourceRegion = np.uint8(self.sourceRegion) # dtype = np.uint8
        self.originalSourceRegion = np.copy(self.sourceRegion)
        
        self.confidence = np.float32(self.confidence)
 
        # _, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        # _, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = self.mask.copy()
        self.targetRegion = np.uint8(self.targetRegion)
        self.data = np.ndarray(shape = self.inputImage.shape[:2],  dtype = np.float32)

        #Initialize 
        self.fillFront = []
        self.normals = []
        self.priorities = []
        
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype = np.float32)
        self.NORMAL_KERNELX[1, 0] = -1
        self.NORMAL_KERNELX[1, 2] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

    def calculateGradients(self):
        srcGray = cv2.cvtColor(self.workImage, cv2.COLOR_BGR2GRAY) # TODO: check type CV_BGR2GRAY
        
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0) # default parameter: scale shoule be 1
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)
        cv2.imwrite("results/gradientX.png", self.gradientX)
        cv2.imwrite("results/gradientY.png", self.gradientY)
        height, width = self.sourceRegion.shape
        #Explicitly set the gradients to zero for the masked area
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    self.gradientX[y, x] = 0
                    self.gradientY[y, x] = 0
        self.gradientX /= 255
        self.gradientY /= 255

    def computeFillFront(self):
        # elements of boundryMat, whose value > 0 are neighbour pixels of target region. 
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)
        _, out = cv2.threshold(boundryMat, 0.1, 255, cv2.THRESH_BINARY)
        cv2.imwrite("results/boundary.png", out.astype('uint8'))
        sourceGradientX = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)
        del self.fillFront[:]
        del self.normals[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y))
                    dx = sourceGradientX[y, x]
                    dy = sourceGradientY[y, x]
                    
                    normalX, normalY = dy, - dx 
                    tempF = math.sqrt(pow(normalX, 2) + pow(normalY, 2))
                    if not tempF == 0:
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))
    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight
    
    def computeConfidence(self):
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)
            total = 0
            for y in range(aY, bY + 1):
                for x in range(aX, bX + 1):
                    if self.targetRegion[y, x] == 0:
                        total += self.confidence[y, x]
            self.confidence[pY, pX] = total / ((bX-aX+1) * (bY-aY+1))
    
    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[y, x] = math.fabs(self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY) + 0.001
    
    def getNeighbours(self, matrix, point, size=3):
        half_size = size //2
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - half_size, 0)
        maxX = min(centerX + half_size, width - 1)
        minY = max(centerY - half_size, 0)
        maxY = min(centerY + half_size, height - 1)
        return matrix[minY:maxY+1, minX:maxX+1]

    
    def computePriority(self):
        self.prior = 0
        maxPriority, priority = 0, 0
        omega, alpha, beta = 0.7, 0.2, 0.8
        
        self.priorities = np.zeros_like(self.confidence, dtype=np.float32)
        #Pixel wise priority term
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            # Way 1
            # priority = self.data[y, x] * self.confidence[y, x]
            # Way 2
            Rcp = (1-omega) * self.confidence[y, x] + omega
            priority = alpha * Rcp + beta * self.data[y, x]
            self.priorities[y,x] = priority
            
            # if priority > maxPriority:
            #     maxPriority = priority
            #     self.targetIndex = i
        for i in range(len(self.fillFront)):
            x,y = self.fillFront[i]
            priority_patch = self.getNeighbours(self.priorities, (x,y))
            priority = priority_patch.sum() / np.count_nonzero(priority_patch)
            if priority > maxPriority:
                maxPriority = priority
                self.prior = i

        

