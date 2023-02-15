import argparse
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2
import imageio
import scipy.ndimage as ndi
import os
from PIL import Image


class GetRegisteredImage:

    def __init__(self, CTscanImage, MRIscanImage):
        self.counter = 0
        self.counterm = 0
        self.CTScanCoords= []
        self.MRIScanCoords = []
        self.targetImageMatrix =[]
        self.inputImageMatrix = []
        # Running the Functions as follows,

        cv2.imshow('CT Scan Image', CTscanImage)
        cv2.setMouseCallback('CT Scan Image', self.clickEventCTscanImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('MRI Scan Image', MRIscanImage)
        cv2.setMouseCallback('MRI Scan Image', self.clickEventMRIscanImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Defining the Click Event Function for CT Scan Image
    def clickEventCTscanImage(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.counter <= 5:
                print(x,y)
                self.CTScanCoords.append([x,y])
                self.counter = self.counter + 1
            
        if self.counter == 5:
            print('[+] CT Scan Points Completed.')
            cv2.destroyAllWindows()
            return 0


    # Defining the Click Event Function for MRI Scan Image
    def clickEventMRIscanImage(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.counterm <= 5:
                print(x,y)
                self.MRIScanCoords.append([x,y])
                self.counterm = self.counterm + 1
        if self.counterm == 5:
            print('[+] MRI Scan Points Completed.')
            cv2.destroyAllWindows()
            return 0


    def checkMatrixIsZero(self, matrix):
        # Iterate through each element in the matrix
        for row in matrix:
            for element in row:
                # If any element is not equal to zero, return False
                if element != 0:
                    return False
        # If all elements are zero, return True
        return True


    # Defining a port of MATLAB's `procrustes` function to Numpy.
    def procrustes(self, targetImageMatrix, inputImageMatrix, scaling=True, reflection='best'):
        rowsInTarget, columnsInTarget = targetImageMatrix.shape # number of rows, no of columns in Target Image
        rowsInInput, columnsInInput = inputImageMatrix.shape # number of rows, no of columns in Input Image
    
        targetMean = targetImageMatrix.mean(0) #mean of each column calculated separtely in Target Image
        inputMean = inputImageMatrix.mean(0) #mean of each column calculated separtely in Input Image

        resultedTargetMatrix = targetImageMatrix - targetMean
        resultedInputMatrix = inputImageMatrix - inputMean
        
        # Checking if the Resulted Matrices having all the elements zero.
        if self.checkMatrixIsZero(resultedTargetMatrix) == True:
            print("All the values(x,y) of CT Image Matrix are same. Hence, Can't process further. Please select different control points!")
            return 0, 0, 0

        if self.checkMatrixIsZero(resultedInputMatrix) == True:
            print("All the values(x,y) of MRI Image Matrix are same. Hence, Can't process further. Please select different control points!")
            return 0, 0, 0

        else:

            # Sum of squares of each element all together for Resulted Target Matrix and Resulted Input Matrix
            sumOfSquaresTarget = (resultedTargetMatrix ** 2.).sum()
            sumOfSquaresInput = (resultedInputMatrix ** 2.).sum()
            print(f'[+] Sum of Sqaures of Target Image: {sumOfSquaresTarget}') #27724.800000000003
            print(f'[+] Sum of Sqaures of Input Image: {sumOfSquaresInput}')
            
            
            # Centred Frobenius Normalization
            normTargetImage = np.sqrt(sumOfSquaresTarget)
            normInputImage = np.sqrt(sumOfSquaresInput)


            # Scaling to equal (unit) normalizaton
            resultedTargetMatrix /= normTargetImage 
            resultedInputMatrix /= normInputImage
            np.seterr(divide='ignore', invalid='ignore')

            if columnsInInput < columnsInTarget:
                resultedInputMatrix = np.concatenate((resultedInputMatrix, np.zeros(rowsInTarget, columnsInTarget - columnsInInput)),0)  # Pads Resulted Input Image to match number of columns in Resulted Target Inage 

    
            # Optimum rotation matrix of Input Image
            A = np.dot(resultedTargetMatrix.T, resultedInputMatrix)
            U,s,Vt = np.linalg.svd(A,full_matrices=False)
            V = Vt.T
            T = np.dot(V, U.T)


            # Optimum Reflection
            if reflection is not 'best':
                
                # does the current solution use a reflection?
                have_reflection = np.linalg.det(T) < 0

                # if that's not what was specified, force another reflection
                if reflection != have_reflection:
                    V[:,-1] *= -1 # Negates the last column of matrix V
                    s[-1] *= -1 # Negates the last element of the array s.
                    T = np.dot(V, U.T) 
 
            traceTA = s.sum()


            # Scaling
            if scaling:
                b = traceTA * normTargetImage / normInputImage # optimum scaling of inputImageMatrix
                d = 1 - traceTA ** 2 # standarised distance between X and b * Y * T + c
                Z = normTargetImage * traceTA * np.dot(resultedInputMatrix, T) + targetMean # transformed coords

            else:
                b = 1
                d = 1 + sumOfSquaresInput/sumOfSquaresTarget - 2 * traceTA * normInputImage / normTargetImage
                Z = normInputImage*np.dot(resultedInputMatrix, T) + targetMean


            # Transformation matrix
            if columnsInInput < columnsInTarget:
                T = T[:columnsInInput,:]
            c = targetMean - b*np.dot(inputMean, T)
            #rot =1
            #scale=2
            #translate=3
            #transformation values 
            tForm = {'rotation':T, 'scale':b, 'translation':c}

            return d, Z, tForm


    # Converting input list into array
    def convertToArray(self, CTScanCoords, MRIScanCoords):
        procrustresReturnList = []
        self.targetImageMatrix = np.asarray(self.CTScanCoords)
        self.inputImageMatrix = np.asarray(self.MRIScanCoords)
        procrustresReturnList = self.procrustes(self.targetImageMatrix, self.inputImageMatrix)
        
        #Checking if the value returned by the procrustes function is not equal to zero. It will return D, Z, tForm
        if any(item != 0 for item in procrustresReturnList):
            return procrustresReturnList[0], procrustresReturnList[1], procrustresReturnList[2]


    def registeringImage(self, d, Z, tForm, CTscanImage):

        R = np.eye(3)
        R[0:2, 0:2] = tForm['rotation']
        S = np.eye(3) * tForm['scale']
        S[2,2] = 1
        t = np.eye(3)
        t[0:2,2] = tForm['translation']
        M = np.dot(np.dot(R,S), t.T).T

        # Height and Width of CT Image
        height = CTscanImage.shape[0]
        width = CTscanImage.shape[1]

        trYImg = cv2.warpAffine(MRIscanImage, M[0:2,:], (height,width))
        cv2.imwrite("jpg/registered_MRI.jpg", trYImg)

        aYPts = np.hstack((self.inputImageMatrix, np.array(([[1,1,1,1,1]])).T))
        trYPts = np.dot(M, aYPts.T).T

        plt.figure() 
        plt.subplot(1,3,1)
        plt.imshow(CTscanImage,cmap=cm.gray)
        plt.plot(self.targetImageMatrix[:,0], self.targetImageMatrix[:,1],'bo',markersize=5)
        # plt.axis('off')
        # plt.subplot(1,3,2)
        # plt.imshow(mri_registered,cmap=cm.gray)
        # plt.plot(Y_pts[:,0],Y_pts[:,1],'ro',markersize=5)
        # plt.axis('off')
        plt.subplot(1,3,3)
        # plt.imshow(ct_fixed,cmap=cm.gray)
        plt.imshow(trYImg,cmap=cm.gray)
        # plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5) 
        # plt.plot(Z_pts[:,0],Z_pts[:,1],'ro',markersize=5)
        plt.plot(trYPts[:,0],trYPts[:,1],'gx',markersize=5)
        # plt.axis('off')
        plt.show()
        


if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--pathInCTImage", help="Path to CT Scan Image")
    arg.add_argument("--pathInMRIImage", help="Path to MRI Scan Image")
    args = arg.parse_args()


    CTscanImage = cv2.imread(args.pathInCTImage)
    MRIscanImage = cv2.imread(args.pathInMRIImage) #MRI Image will be registered

    handler = GetRegisteredImage(CTscanImage, MRIscanImage)


    print(f'[+] CT Scan Codes: {handler.CTScanCoords}')
    print(f'[+] MRI Scan Codes: {handler.MRIScanCoords}')

    d, Z, tForm = handler.convertToArray(handler.CTScanCoords, handler.MRIScanCoords)

    handler.registeringImage(d, Z, tForm, CTscanImage)




