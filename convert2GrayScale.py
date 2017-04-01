import numpy as np
import scipy.misc
from PIL import Image
import cPickle

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
    """
    cond = img == 1
    img[cond] =  0
    return img

def getDistinctLabels(img):
    img = img.reshape(-1,img.shape[-1])
    img = map(tuple,img)
    img = sorted(list(set(img)))
    ret = []
    for i in range(len(img)):
        ret.append((img[i],i))
    print ret
    return ret

def getGreyScaleImage(img,labels):
    res = np.zeros((img.shape[0],img.shape[1]),img.dtype)
    for label in labels:
        cond = (img[:,:,0] == label[0][0]) & (img[:,:,1] == label[0][1]) &( img[:,:,2] == label[0][2])
        res[cond] = label[1]
    return res

def imsave(path, img):
    #img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

def main():
    """
    get distinct colors from the image
    sort them by their rgb
    and assign them one grey scale value
    Obviously this wont work a good color image
    this code is only for segmented images
    which should have like one to 16 colors
    """
    img = imread('../deep/testImages/styleImageSegmentation.jpg')
    labels =  getDistinctLabels(img)
    print labels
    gImg = getGreyScaleImage(img,labels)
    cPickle.dump(gImg,open('styleSegmentation.pickle','wb'))



if __name__ == '__main__':
    main()
