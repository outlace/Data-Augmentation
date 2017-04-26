import numpy as np
from scipy import ndimage

# NOTE
# Images are assumed to be uint8 0-255 valued.
# For augment function:
#   images shape: (batch_size, height, width, channels=3)
#   labels shape: (batch_size, 3)

def addBlotch(image, max_dims=[0.2,0.2]):
    #add's small black/white box randomly in periphery of image
    new_image = np.copy(image)
    shape = new_image.shape
    max_x = shape[0] * max_dims[0]
    max_y = shape[1] * max_dims[1]
    rand_x = 0
    rand_y = np.random.randint(low=0, high=shape[1])
    rand_bool = np.random.randint(0,2)
    if rand_bool == 0:
        rand_x = np.random.randint(low=0, high=max_x)
    else:
        rand_x = np.random.randint(low=(shape[0]-max_x), high=shape[0])
    size = np.random.randint(low=1, high=7) #size of each side of box
    new_image[rand_x:(size+rand_x), rand_y:(size+rand_y), :] = np.random.randint(0,256)
    return new_image

def shift(image, max_amt=0.2):
    new_img = np.copy(image)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)
    return ndimage.interpolation.shift(new_img,shift=[x,y,0])

def addNoise(image, amt=0.005):
    noise_mask = np.random.poisson(image / 255.0 * amt) / amt * 255
    noisy_img = image + (noise_mask)
    return np.array(np.clip(noisy_img, a_min=0., a_max=255.), dtype=np.uint8)

def rotate(image):
    randnum = np.random.randint(1,360)
    new_image = np.copy(image)
    return ndimage.rotate(new_image, angle=randnum, reshape=False)

#randomly manipulates image
#rotate, flip along axis, add blotch, shift
def augment(images, labels=None, amplify=2):
    # INPUT:
    #images shape: (batch_size, height, width, channels=3)
    #labels shape: (batch_size, 3)
    ops = {
        0: addBlotch,
        1: shift,
        2: addNoise,
        3: rotate
    }

    shape = images.shape
    new_images = np.zeros(((amplify*shape[0]), shape[1], shape[2], shape[3]))
    if labels is not None:
        new_labels = np.zeros(((amplify*shape[0]), 3))
    for i in range(images.shape[0]):
        cur_img = np.copy(images[i])
        new_images[i] = cur_img
        if labels is not None:
            new_labels[i] = np.copy(labels[i])
        for j in range(1, amplify):
            add_r = ( j * shape[0] )
            which_op = np.random.randint(low=0, high=4)
            dup_img = np.zeros((1,shape[1], shape[2], shape[3]))
            new_images[i+add_r] = ops[which_op](cur_img)
            if labels is not None:
                new_labels[i+add_r] = np.copy(labels[i])
    if labels is not None:
        return new_images.astype(np.uint8), new_labels.astype(np.uint8)
    else:
        return new_images.astype(np.uint8)
