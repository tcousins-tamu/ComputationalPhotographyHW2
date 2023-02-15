# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import math

#import TiffImagePlugin

#Autocropping function
def auto_crop(img):
    '''
    We know that the colors are going to (roughly) be within the ranges specified below.
    The goal then, is to find the square that best fits all of these values

    Status: Should be good for now, show EMIR picture as proof
    '''
    #The target colors are the color ranges we anticipate the borders being in
    targetColors = np.asarray([
        #The values below are NOT normallized to 1
        # [ [255, 255, 255], [200,200,200]], #White
        # [ [50,50,50] ,[0, 0, 0] ], #Black
        # [ [255, 50, 50],[200,0,0]],  #Red
        # [ [50, 255, 50], [0,200,0]], #Blue
        # [ [50, 50, 255], [0,0,200]] #Green
        #These are
        [ [1, 1, 1], [.78,.78,.78]], #White
        [ [.2,.2,.2] ,[0, 0, 0] ], #Black
        [ [1, .2, .2],[.78,0,0]],  #Red
        [ [.2, 1, .2], [0,.78,0]], #Blue
        [ [.2, .2, 1], [0,0,.78]] #Green
    ])

    #Iterating through each target color that falls within these bounds
    r = img[:,:, 0]
    g = img[:,:, 1]
    b = img[:,:, 2]
    bnds = np.full(r.shape, False)
    for upperBound, lowerBound in targetColors:
        #Each target color needs to have the r, g, and b values in bounds
        cBnds = np.where((r<=upperBound[0]) & (r>=lowerBound[0]) & (b<=upperBound[1]) & (b>=lowerBound[1]) & (g<=upperBound[2]) & (g>=lowerBound[2]))
        bnds[cBnds] = True
    
    #Now we have a pixel wise map of where colors in this boundary are, the next step is going to be to find a well-fitting rectangle
    xBounds = [0, r.shape[0]]
    yBounds = [0, r.shape[1]]

    #starting from lowest y, we will find a good y value
    startingSum = 0
    tolerance = 50 #we know that when a lot more points start failing the test, the edge is gone
    for lowY in range(0,bnds.shape[0]):
        borderSum = bnds[lowY, :].sum() #Finds the number of points that are considered potential edges at this case
        if borderSum>startingSum:
            startingSum = borderSum
        elif borderSum<(startingSum-tolerance):
            yBounds[0] = lowY-1
            break
    
    startingSum = 0
    for highY in reversed(range(0, bnds.shape[0])):
        borderSum = bnds[highY, :].sum() #Finds the number of points that are considered potential edges at this case
        if borderSum>startingSum:
            startingSum = borderSum
        elif borderSum<(startingSum-tolerance):
            yBounds[1] = highY+1
            break
    
    startingSum = 0
    for lowX in range(0,bnds.shape[1]):
        borderSum = bnds[:, lowX].sum() #Finds the number of points that are considered potential edges at this case
        if borderSum>startingSum:
            startingSum = borderSum
        elif borderSum<(startingSum-tolerance):
            xBounds[0] = lowX-1
            break

    startingSum = 0
    for highX in reversed(range(0, bnds.shape[1])):
        borderSum = bnds[:, highX].sum() #Finds the number of points that are considered potential edges at this case
        if borderSum>startingSum:
            startingSum = borderSum
        elif borderSum<(startingSum-tolerance):
            xBounds[1] = highX+1
            break

    return img[lowX:highX, lowY:highY, :]

def rotate(src_img, angle_of_rotation, pivot_point, shape_img):

    #1.create rotation matrix with numpy array
    rotation_mat = np.transpose(np.array([[np.cos(angle_of_rotation),-np.sin(angle_of_rotation)],
                            [np.sin(angle_of_rotation),np.cos(angle_of_rotation)]]))
    h,w = shape_img
    
    pivot_point_x =  pivot_point[0]
    pivot_point_y = pivot_point[1]
    
    new_img = np.zeros(src_img.shape,dtype='u1') 

    for height in range(h): #h = number of row
        for width in range(w): #w = number of col
            xy_mat = np.array([[width-pivot_point_x],[height-pivot_point_y]])
            
            rotate_mat = np.dot(rotation_mat,xy_mat)

            new_x = pivot_point_x + int(rotate_mat[0])
            new_y = pivot_point_y + int(rotate_mat[1])


            if (0<=new_x<=w-1) and (0<=new_y<=h-1): 
                new_img[new_y,new_x] = src_img[height,width]

    return new_img

def auto_contrast(img, mode="gauss", **kwargs):
    '''
    This function will remap the values to a non-linear mapping
    '''
    output = img
    if mode == 'gauss':
        center = .5
        height = 1
        stdev = .2
        
        if "center" in kwargs:
            center = float(kwargs["mean"])
        if "stdev" in kwargs:
            stdev = float(kwargs["cov"])
        if "height" in kwargs:
            height = float(kwargs["cov"])

        output = height * math.e ** (-((img - center)**2)/(2*stdev**2))
    
    elif mode== 'log':
        lower = 0
        upper = 1
        
        if "lower" in kwargs:
            lower = float(kwargs["mean"])
        if "upper" in kwargs:
            upper = float(kwargs["cov"])
        
        output = np.clip(3.3220 * np.log10(img+1), 0, 1)
        
    return output

def auto_whiteBalance(img):
    '''
    White balances the image
    '''
    #Step One, obtaining the magnitude to determine where the likely source of the illuminant is
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    mag = r+g+b #array containing the magnitudes, we will use a standard weighting here 

    #We do not necessarily need to estimate the exact location of the illuminant, so long as we take an aribtrary amount of the brightest values and scale off of their mean
    numItems = r.shape[0]*r.shape[1]
    illPcnt = .1 #percent that is illuminated points 105
    illPts = int(illPcnt*numItems)

    brightPts = np.argpartition(mag.flatten(), -1*illPts)[-1*illPts:] #These are the locations of the illPts number of brightest points
    print(brightPts)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()
    rMean = np.mean(r[brightPts])
    gMean = np.mean(g[brightPts])
    bMean = np.mean(b[brightPts])
    
    #we want the average mean to approach the mean of the ten percent brightest
    brtNot = np.full(r.shape, True)
    brtNot[brightPts] = False

    #Average brightness amongst the other points
    rnMean = np.mean(r[brtNot])
    bnMean = np.mean(r[brtNot])
    gnMean = np.mean(r[brtNot])

    #we want the average brightness to approach that of the brightest, but not be quite as bright
    r[brtNot] = .7*r[brtNot]*(rMean/rnMean)
    b[brtNot] = .7*b[brtNot]*(bMean/bnMean)
    g[brtNot] = .7*g[brtNot]*(gMean/gnMean)

    #we will now scale using these brightest points
    # maxR = np.max(r)
    # maxB = np.max(b)
    # maxG = np.max(g)

    #Linear interpolation to new scale

    img[:, :, 0] = np.reshape(np.clip(r, 0, .9), img.shape[:-1])
    img[:, :, 1] = np.reshape(np.clip(g, 0, .9), img.shape[:-1])
    img[:, :, 2] = np.reshape(np.clip(b, 0, .9), img.shape[:-1])

    return img

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
def find_shift(im1, im2, xShift = 20, yShift=20,  mode = "RGB"):
    '''
    Will return optimal shift to align im1 TO im2. 
    Parameters xShift and yShift are bounds on how far it will attempt to shift.
    '''
    xOpt, yOpt = 0,0 #xOptimal and y Optimal
    minSum = np.inf #minimization base

    #Makes our shifts always positive, speeds up calcs
    im1Bounded = circ_shift(im1, [-xShift, -yShift])
    if mode == "RGB":
        for x in range(xShift*2):
            for y in range(yShift*2):
                sumT = np.sum((im2[x:,y:]-circ_shift(im1Bounded,[x, y])[x:,y:])**2)
                if sumT<minSum:
                    minSum=sumT
                    xOpt, yOpt = x, y
        return xOpt-xShift, yOpt-yShift

    #Better transforms will work the same as above but have a case for rotation and scaling
    #Because we want this to fit in or current implementation, we want to rotate and scale the actual datasets
    elif mode == "bT":
        imOut = im1Bounded
        for x in range(xShift*2):
            for y in range(yShift*2):
                #We will only be doing a coarse check b/c <10* is not really noticable
                for theta in np.linspace(-math.pi, math.pi, 36):
                    #Scalar value, this is also pretty coarse for the same reasons
                    for scale in reversed(np.linspace(.5, 1, 10)):
                        #If we find a scalar that makes the values better, we will apply it
                        print(im1Bounded.shape)
                        im1Mod = resize(im1Bounded, [int(im1Bounded.shape[0]*scale), int(im1Bounded.shape[1]*scale)])
                        #Padding the values such that the shape remains the same after resizing
                        ax1Pad = [np.floor(im1Bounded.shape[0]-im1Mod.shape[0]), np.ceil(im1Bounded.shape[0]-im1Mod.shape[0])]
                        ax2Pad = [np.floor(im1Bounded.shape[1]-im1Mod.shape[1]), np.ceil(im1Bounded.shape[1]-im1Mod.shape[1])]
                        im1Mod = np.pad(im1Mod, [ax1Pad, ax2Pad])

                        #Rotating the array
                        im1Mod = rotate(im1Mod, theta, [int(im1Bounded.shape[0]/2), int(im1Bounded.shape[1]/2)], im1Mod.shape)
    
                        sumT = np.sum((im2[x:,y:]-circ_shift(im1Mod,[x, y])[x:,y:])**2)
                        if sumT<minSum:
                            imOut = im1Mod
                            minSum = sumT
                            xOpt, yOpt = x, y
                            #Finer Search is Done here
                            for thetaFine in (np.linspace(theta - 0.1745, theta + 0.1745, 20)):
                                for scaleFine in (np.linspace(scale-.1, scale+.1, 5)):

                                    im1Mod = resize(im1Bounded, [int(im1Bounded.shape[0]*scaleFine), int(im1Bounded.shape[1]*scaleFine)])
                                    #Padding the values such that the shape remains the same after resizing
                                    ax1Pad = [np.floor(im1Bounded.shape[0]-im1Mod.shape[0]), np.ceil(im1Bounded.shape[0]-im1Mod.shape[0])]
                                    ax2Pad = [np.floor(im1Bounded.shape[1]-im1Mod.shape[1]), np.ceil(im1Bounded.shape[1]-im1Mod.shape[1])]
                                    im1Mod = np.pad(im1Mod, [ax1Pad, ax2Pad])

                                    im1Mod = rotate(im1Mod, thetaFine, [int(im1Bounded.shape[0]/2), int(im1Bounded.shape[1]/2)], im1Mod.shape)
                                    sumT = np.sum((im2[x:,y:]-circ_shift(im1Mod,[x, y])[x:,y:])**2)

                                    if sumTFine<minSum:
                                        imOut = im1Mod
                                        minSum = sumT
                                        xOpt, yOpt = x, y
        im1 = circ_shift(imOut, [xShift, yShift])
        return xOpt-xShift, yOpt-yShift

    elif mode=="edges":
        #This implementation is similar to above, however it uses the edges generated by roughly the partial derivatives
        #Need to normalize values
        im1n = im1Bounded-np.min(im1Bounded)
        im1n = im1n*(1/np.max(im1n))

        im2n = im2-np.min(im2)
        im2n = im2n*(1/np.max(im2n))
        #It then sets a floor to remove noise values
        edg1X = (np.diff(im1n, axis=-1))
        edg1Y = (np.diff(im1n, axis=0))
        edg2X = (np.diff(im2n, axis=-1))
        edg2Y = (np.diff(im2n, axis=0))

        #Removing low values MODIFY THIS FOR LADY CASE
        edg1X[np.where((edg1X<.1) & (edg1X>-.1))]=0
        edg1Y[np.where((edg1Y<.1) & (edg1Y>-.1))]=0
        edg2X[np.where((edg2X<.1) & (edg2X>-.1))]=0
        edg2Y[np.where((edg2Y<.1) & (edg2Y>-.1))]=0

        #weighting all edges equivalently, this is b/c a jump in blue could be a drop in red
        edg1X[np.where(edg1X>=.1)]=1
        edg1X[np.where(edg1X<=-.1)]=1

        edg1Y[np.where(edg1Y>=.1)]=1
        edg1Y[np.where(edg1Y<=-.1)]=1
        
        edg2X[np.where(edg2X>=.1)]=1
        edg2X[np.where(edg2X<=-.1)]=1
        
        edg2Y[np.where(edg2Y>=.1)]=1
        edg2Y[np.where(edg2Y<=-.1)]=1

        #Now we can do sum of squared once again
        for x in range(xShift*2):
            for y in range(yShift*2):
                sumX = np.sum((edg2X[x:,y:]-circ_shift(edg1X,[x, y])[x:,y:])**2)
                sumY = np.sum((edg2Y[x:,y:]-circ_shift(edg1Y,[x, y])[x:,y:])**2)
                sumT = sumX+sumY
                if sumT<minSum:
                    minSum=sumT
                    xOpt, yOpt = x, y #One is added b/c diff reduces dimensionality by 1
        return xOpt-xShift, yOpt-yShift

    #Neerd to implement a case that matches edges better. Would like to detect edes by 
    #Will take partial derivative with respect to x and y then do the shifting like that

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
    #output
    finalImage = '\0'
    translation = []

    for imageName in os.listdir(imageDir):
        # Get r, g, b channels from image strip
        r, g, b = read_strip(imageDir + imageName)
        
        #Specifying bounds for x and y shift
        xShift = 20
        yShift = 20

        mode = 'bT'
        #Assignment 1, modifying the jpg files
        if os.path.splitext(imageName)[-1] == '.jpg':
            # Calculate shift
            rShift = find_shift(r, b, mode=mode)
            gShift = find_shift(g, b, mode=mode)
            # Shifting the images using the obtained shift values
            finalB = b
            finalG = circ_shift(g, gShift)
            finalR = circ_shift(r, rShift)

            translation = [rShift, gShift]
            # Putting together the aligned channels to form the color image
            finalImage = np.stack((finalR, finalG, finalB), axis = 2)

            # Writing the image to the Results folder
            #plt.imsave(outDir + imageName[:-4] + '.jpg', finalImage)
        
        elif os.path.splitext(imageName)[-1] == '.tif':
            translation = np.zeros([2,2], int) #[[0,0], [0,0]] #rShift, gShift
            baseImg = np.asarray([r,g,b]) #Base Data
            shape = np.asarray([3,(baseImg.shape[1]/(2**(levels-1))), (baseImg.shape[2]/(2**(levels-1)))]) #Size of smallest level

            for step in (range(0,levels)):
                imScaled = resize(baseImg, shape.astype(int)) #cast to int for odd numbers
                rS, gS, bS = imScaled
                rS = circ_shift(rS, translation[0])
                gS = circ_shift(gS, translation[1])
                
                #Calculate Shift
                #We do not need to continue to search the same space as it gets larger
                rShift = np.asarray(find_shift(rS, bS, xShift, yShift, mode=mode))
                gShift = np.asarray(find_shift(gS, bS, xShift, yShift, mode=mode))
                translation = np.add(translation, [rShift, gShift])

                #For each iteration, the area of the pyramid doubles
                translation = translation*2 #This means the translation doubles
                shape[1:] = shape[1:]*2 #The shape doubles
                xShift = int(xShift/2) #The xShift can half
                yShift = int(yShift/2) #The Y shift can half
                
                if xShift<1:
                    xShift = 1
                if yShift<1:
                    yShift = 1
            
            #Compiling the final image
            translation = np.asarray(translation/2, int)
            finalB = baseImg[2]
            finalG = circ_shift(baseImg[1], translation[1])
            finalR = circ_shift(baseImg[0], translation[0])
            finalImage = np.stack((finalR, finalG, finalB), axis = 2)
        
        #print(finalImage.shape)
        #finalImage = auto_crop(finalImage) #Has some issues with some points, but works well on emir
        #auto_whiteBalance(finalImage) #Currently makes things neutrally bright, so for centralized brightness it may look funny
        #finalImage = auto_contrast(finalImage, "log")
        # X = np.abs(np.diff(finalImage, axis = 0))
        # X = X-np.min(X)
        # X = X*(1/(np.max(X)))
        # X[np.where((X<.1) & (X>-.1))] = 0
        # X[np.where((X>=.1) | (X<=-.1))] = 1

        # Y = np.abs(np.diff(finalImage, axis = 1))
        # Y = Y-np.min(Y)
        # Y = Y*(1/np.max(Y))
        # Y[np.where((Y<.1) & (Y>-.1))] = 0
        # Y[np.where((Y>=.1) | (Y<=-.1))] = 1


        # plt.imsave(outDir + imageName[:-4] + 'X.tiff', X)
        # plt.imsave(outDir + imageName[:-4] + 'Y.tiff', Y)  
        print("###########################################################")
        print(imageName, r.shape, "GShift: ",translation[1], "rShift: ", translation[0], "translation: \n", translation)

        ext = os.path.splitext(imageName)[-1]
        if ext == '.tif':
            ext = '.tiff'
        plt.imsave(outDir + imageName[:-4] + ext, finalImage)
