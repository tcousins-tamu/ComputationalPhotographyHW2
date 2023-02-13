# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path) # read the input image
    info = np.iinfo(image.dtype) # get information about the image type (min max values)
    image = image.astype(float) / info.max # normalize the image into range 0 and 1

    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535
    
    # Separating the glass image into R, G, and B channels
    b = image[: height, :]
    g = image[height: 2 * height, :]
    r = image[2 * height: 3 * height, :]
    return r, g, b

# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0)
    shifted = np.roll(shifted, shift[1], axis = 1)
    return shifted

# The main part of the code. Implement the FindShift function
def find_shift(im1, im2, xShift = 20, yShift=20):
    '''
    Will return optimal shift to align im1 TO im2. 
    Parameters xShift and yShift are bounds on how far it will attempt to shift.
    '''
    xOpt, yOpt = 0,0 #xOptimal and y Optimal
    minSum = np.inf #minimization base

    #Makes our shifts always positive, speeds up calcs
    im1Bounded = circ_shift(im1, [-xShift, -yShift])
    for x in range(xShift*2):
        for y in range(yShift*2):
            sum = np.sum((im2[x:,y:]-circ_shift(im1Bounded,[x, y])[x:,y:])**2)
            if sum<minSum:
                minSum=sum
                xOpt, yOpt = x, y
    return xOpt-xShift, yOpt-yShift

    #Implementation 1, no care for outliers
    # for x in range(xShift*2):
    #     for y in range(yShift*2):
    #         sum = np.sum((im2-circ_shift(im1,[x, y]))**2)
    #         if sum<minSum:
    #             minSum=sum
    #             xOpt, yOpt = x, y
    # return xOpt, yOpt

if __name__ == '__main__':
    # Setting the input output file path
    imageDir = './Images/'
    #imageName = 'turkmen.tif'
    outDir = './Results/'
    #Specifying the number of levels in the image pyramid
    levels = 4 
    
    for imageName in os.listdir(imageDir):
        # Get r, g, b channels from image strip
        r, g, b = read_strip(imageDir + imageName)

        #Assignment 1, modifying the jpg files
        if os.path.splitext(imageName)[-1] == '.jpg':
            continue
            # Calculate shift
            rShift = find_shift(r, b)
            gShift = find_shift(g, b)

            print(imageName, r.shape, "GShift: ",gShift, "rShift: ", rShift)
            # Shifting the images using the obtained shift values
            finalB = b
            finalG = circ_shift(g, gShift)
            finalR = circ_shift(r, rShift)

            # Putting together the aligned channels to form the color image
            finalImage = np.stack((finalR, finalG, finalB), axis = 2)

            # Writing the image to the Results folder
            plt.imsave(outDir + imageName[:-4] + '.jpg', finalImage)
        
        elif os.path.splitext(imageName)[-1] == '.tif':
            #Will need to scale downm
            #Loop Parameters
            translation = [np.asarray([0,0]), np.asarray([0,0])] #rShift, gShift
            baseImg = np.asarray([r,g,b]) #Base Data
            shape = np.asarray([3,(baseImg.shape[1]/(2**(levels-1))), (baseImg.shape[2]/(2**(levels-1)))]) #Size of smallest level

            #get it in a range from 4->1
            #for scale in reversed(range(levels+1)[1:]):
            for step in range(levels-1):
                imScaled = resize(baseImg, shape.astype(int)) #cast to int for odd numbers
                rS, gS, bS = imScaled
                rS = circ_shift(rS, translation[0])
                gS = circ_shift(gS, translation[1])
                
                #Calculate Shift
                rShift = np.asarray(find_shift(rS, bS))
                gShift = np.asarray(find_shift(gS, bS))
                translation[0]=translation[0]+rShift
                translation[1]=translation[1]+gShift
                # translation = np.asarray([[translation[0]+find_shift(rS, bS)],[translation[1]+find_shift(gS, bS)]])
                print(translation)
                print(imageName, rS.shape, "GShift: ",gShift, "rShift: ", rShift)

                #For each iteration, the area of the pyramid doubles
                translation=translation*2
                shape[1:] = shape[1:]*2
            
            #Compiling the final image
            finalB = baseImg[2]
            finalG = circ_shift(baseImg[1], translation[1])
            finalR = circ_shift(baseImg[0], translation[0])
            finalImage = np.stack((finalR, finalG, finalB), axis = 2)

            plt.imsave(outDir + imageName[:-4] + '.tif', finalImage)
