import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
import matplotlib.pyplot as plt
import pywt
import pywt.data

# Defining a custom pre-trained model (VGG19)
class VGG19(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()
        modelFeatures = list(vgg19(pretrained=True).features) # Extracting the feature of the model
        if device == "cuda":
            self.features = nn.ModuleList(modelFeatures).cuda().eval()
        else:
            self.features = nn.ModuleList(modelFeatures).eval()

    def forward(self, image):
        featureMaps = [] # List used to store the output of the each layer of the VGG19 model
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index == 3:
                featureMaps.append(image)
        return featureMaps




class GetFusedImage:

    def __init__(self, CTscanImage, MRIscanImage):
        """
        Class Fusion constructor

        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.inputImages = CTscanImage, MRIscanImage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGG19(self.device)



    def _RGB_to_YCbCr(self, img_RGB):
            """
            A private method which converts an RGB image to YCrCb format
            """
            img_RGB = img_RGB.astype(np.float32) / 255.
            return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    def _YCbCr_to_RGB(self, img_YCbCr):
            """
            A private method which converts a YCrCb image to RGB format
            """
            img_YCbCr = img_YCbCr.astype(np.float32)
            return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def _is_gray(self, img):
            """
            A private method which returns True if image is gray, otherwise False
            """
            if len(img.shape) < 3:
                return True
            if img.shape[2] == 1:
                return True
            b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
            if (b == g).all() and (b == r).all():
                return True
            return False

    def _softmax(self, tensor):
            """
            A private method which compute softmax ouput of a given tensor
            """
            tensor = torch.exp(tensor)
            tensor = tensor / tensor.sum(dim=1, keepdim=True)
            return tensor

    def _tranfer_to_tensor(self):
            """
            A private method to transfer all input images to PyTorch tensors
            """
            self.images_to_tensors = []
            for image in self.normalizedImages:
                np_input = image.astype(np.float32)
                if np_input.ndim == 2:
                    np_input = np.repeat(np_input[None, None], 3, axis=1)
                else:
                    np_input = np.transpose(np_input, (2, 0, 1))[None]
                if self.device == "cuda":
                    self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
                else:
                    self.images_to_tensors.append(torch.from_numpy(np_input))




    def fuseAlgorithm(self):
        """
        Perform fusion algorithm
        """
        with torch.no_grad():

            imageSumMaps = [-1 for tensor_img in self.images_to_tensors]
            for idx, tensor_img in enumerate(self.images_to_tensors):
                imageSumMaps[idx] = []
                featureMaps = self.model(tensor_img)
                for featureMap in featureMaps:
                    sumMap = torch.sum(featureMap, dim=1, keepdim=True)
                    imageSumMaps[idx].append(sumMap)

            maxFusion = None

            for sumMaps in zip(*imageSumMaps):
            
                features = torch.cat(sumMaps, dim=1)
                weights = self._softmax(F.interpolate(features,
                                        size=self.images_to_tensors[0].shape[2:]))
                weights = F.interpolate(weights,
                                        size=self.images_to_tensors[0].shape[2:])
                currentFusion = torch.zeros(self.images_to_tensors[0].shape)
                for index, tensor_img in enumerate(self.images_to_tensors):
                    currentFusion += tensor_img * weights[:,index]
                if maxFusion is None:
                    maxFusion = currentFusion
                else:
                    maxFusion = torch.max(maxFusion, currentFusion)

            output = np.squeeze(maxFusion.cpu().numpy())
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return output




    def fuseImage(self):
        """
        A top level method which fuse self.images
        """
        # Convert all images to YCbCr format
        self.normalizedImages = [-1 for img in self.inputImages]
        self.YCbCrImages = [-1 for img in self.inputImages]

        #checking if the image is grayscale or not
        for index, img in enumerate(self.inputImages):
            if not self._is_gray(img):
                self.YCbCrImages[index] = self._RGB_to_YCbCr(img)
                self.normalizedImages[index] = self.YCbCrImages[index][:, :, 0] #storingthe first channel of YCbCr image
            else:
                self.normalizedImages[index] = img / 255.
        

        # Transfer all images to PyTorch tensors (arrays)
        self._tranfer_to_tensor()

        # Perform fuse strategy
        fusedImage = self.fuseAlgorithm()[:, :, 0]

        # Reconstruct fused image given rgb input images
        for index, img in enumerate(self.inputImages):
            if not self._is_gray(img):
                self.YCbCrImages[index][:, :, 0] = fusedImage
                fusedImage = self._YCbCr_to_RGB(self.YCbCr_images[idx])
                fusedImage = np.clip(fusedImage, 0, 1)

        return (fusedImage * 255).astype(np.uint8)
        # return fused_img



    def waveletTransformation(self, image, imageName):

    	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    	titles = ['Approximation', ' Horizontal detail',
    	          'Vertical detail', 'Diagonal detail']
    	coeffs2 = pywt.dwt2(image, 'haar')

    	LL, (LH, HL, HH) = coeffs2

    	fig = plt.figure(figsize=(12, 3))
    	for i, a in enumerate([LL, LH, HL, HH]):
    	    ax = fig.add_subplot(1, 4, i + 1)
    	    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

    	    if "CTscanImage" in imageName:
    	    	path='/Users/jaskiratsingh/Desktop/registration_fusion/jpg/ct_'+str(i)+'.jpg'
    	    	cv2.imwrite(path,a)

    	    else:
    	    	path='/Users/jaskiratsingh/Desktop/registration_fusion/jpg/mri_'+str(i)+'.jpg'
    	    	cv2.imwrite(path,a)

    	    ax.set_title(titles[i], fontsize=10)
    	    ax.set_xticks([])
    	    ax.set_yticks([])

    	fig.tight_layout()
    	plt.show()



        
       