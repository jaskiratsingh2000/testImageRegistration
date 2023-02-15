import fusion
import os
import argparse
import cv2
import pywt
import pywt.data



def perform_fusion(pathInCTImage, pathInMRIImage, pathOfCTImageDir, pathOfMRIImageDir):

    CTscanImage = cv2.imread(pathInCTImage)
    MRIscanImage = cv2.imread(pathInMRIImage) 

    handler = fusion.GetFusedImage(CTscanImage, MRIscanImage)

    handler.waveletTransformation(CTscanImage, "CTscanImage")
    handler.waveletTransformation(MRIscanImage, "MRIscanImage")

    listOfCTBanImages = os.listdir(pathOfCTImageDir)
    listOfMRIBanImages = os.listdir(pathOfMRIImageDir)

    for i in range(4):
        CTBand = cv2.imread(os.path.join(pathOfCTImageDir, 'ct_{}.jpg'.format(i)))
        CTBand = cv2.cvtColor(CTBand, cv2.COLOR_BGR2GRAY)

        MRIBand = cv2.imread(os.path.join(pathOfMRIImageDir, 'mri_{}.jpg'.format(i)))
        MRIBand = cv2.cvtColor(MRIBand, cv2.COLOR_BGR2GRAY)

        obj = fusion.GetFusedImage(CTBand, MRIBand)

        fusionImage = obj.fuseImage()

        cv2.imwrite('/Users/jaskiratsingh/Desktop/registration_fusion/jpg/fusion_{}.jpg'.format(i), fusionImage)


def reconstructInverseWaveletTransform(pathOfCTImageDir):

    fusionImages = []

    for i in range(4):

        fusion = cv2.imread(os.path.join(pathOfCTImageDir, 'fusion_{}.jpg'.format(i)))
        fusion = cv2.cvtColor(fusion, cv2.COLOR_BGR2GRAY)
        fusionImages.append(fusion)

    coeffs = (fusionImages[0],(fusionImages[1], fusionImages[2], fusionImages[3]))
    fusion = pywt.idwt2(coeffs, 'haar')
    cv2.imwrite('/Users/jaskiratsingh/Desktop/registration_fusion/jpg/final_fusion.jpg', fusion)




if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--pathInCTImage", help="Path to CT Scan Image")
    arg.add_argument("--pathInMRIImage", help="Path to MRI Scan Image")
    arg.add_argument("--pathOfCTImageDir", help="Path to CT Image folder which contains the 4 different bands")
    arg.add_argument("--pathOfMRIImageDir", help="Path to MRI Image folder which contains the 4 different bands")
    args = arg.parse_args()

    perform_fusion(args.pathInCTImage, args.pathInMRIImage, args.pathOfCTImageDir, args.pathOfMRIImageDir)

    reconstructInverseWaveletTransform(args.pathOfCTImageDir)

