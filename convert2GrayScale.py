import numpy as np
import scipy.misc
from PIL import Image
import cPickle
import webcolors


#acceptedColors = ['white','green','blue','red']
acceptedColors = ['black','grey']
acceptedColorsDI = dict(zip(acceptedColors,range(len(acceptedColors))))

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
	if name not in acceptedColors:
		continue
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    closest_name = closest_colour(requested_colour)
    return acceptedColorsDI[closest_name]


def imread(path):
    img = scipy.misc.imread(path)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]

    """
        This code is for styleImageSegmentation.jpg
        which has a bug please remove it for other images
    cond = img  <= 128
    img[cond] =  0
    img[cond==False] = 1
    """
    return img

#def getDistinctLabels(img):
#    """
#    Tmp fix
#    return [((0, 0, 0), 0), ((1, 0, 0), 0), ((1, 1, 0), 0), ((1, 1, 1), 1)]
#    """

def getDistinctLabels(img):
    img = img.reshape(-1,img.shape[-1])
    img = map(tuple,img)
    img = sorted(list(set(img)))
    img = map(lambda l:(l,get_colour_name(l)),img)
    return img

def getGreyScaleImage(img,labels):
    res = np.zeros((img.shape[0],img.shape[1]),img.dtype)
    for label in labels:
        cond = (img[:,:,0] == label[0][0]) & (img[:,:,1] == label[0][1]) &( img[:,:,2] == label[0][2])
        res[cond] = label[1]
    return res

def imsave(path, img):
    #img = np.clip(img, 0, 255).astype(np.uint8)
    print path
    Image.fromarray(img).convert('RGB').save(path, quality=95)

def saveDifferentSegments(img):
    colors = np.unique(img)
    for col in colors:
	cond = img == col
	zero = np.zeros(img.shape)
	zero[cond] = 255
	imsave('seg_{}.png'.format(col),zero)

def main():
    """
    get distinct colors from the image
    sort them by their rgb
    and assign them one grey scale value
    Obviously this wont work a good color image
    this code is only for segmented images
    which should have like one to 16 colors
    """
    img = imread('o.jpg')
    labels =  getDistinctLabels(img)
    gImg = getGreyScaleImage(img,labels)
    cPickle.dump(gImg,open('seth_anuj.pickle','wb'))
    saveDifferentSegments(gImg)



if __name__ == '__main__':
    main()
