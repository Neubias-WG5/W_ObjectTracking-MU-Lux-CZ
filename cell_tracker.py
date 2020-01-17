import numpy as np
import os
import cv2
import tensorflow as tf

from skimage.morphology import watershed

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam


def hist_equalization(image):
    return cv2.equalizeHist(image) / 255

def remove_edge_cells(label_img):
    edge_indexes = get_edge_indexes(label_img)
    return remove_indexed_cells(label_img, edge_indexes)

def get_edge_indexes(label_img, border=1):
    mask = np.ones(label_img.shape) 
    mi, ni = mask.shape
    mask[border:mi-border,border:ni-border] = 0
    border_cells = mask * label_img
    indexes = (np.unique(border_cells))

    result = []

    # get only cells with center inside the mask
    for index in indexes:
        cell_size = sum(sum(label_img == index))
        gap_size = sum(sum(border_cells == index))
        if cell_size * 0.5 < gap_size:
            result.append(index)
    
    return result

def remove_indexed_cells(label_img, indexes):
    mask = np.ones(label_img.shape)
    for i in indexes:
        mask -= (label_img == i)
    return label_img * mask


def get_image_size(path):
    '''
    returns size of the given image
    '''
    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = np.minimum(img, 255).astype(np.uint8)
    return img


# read images
def load_images(path, new_mi=0, new_ni=0):

    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    total = len(names)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    for i, name in enumerate(names):

        o = read_image(os.path.join(path, name))

        if o is None:
            print('image {} was not loaded'.format(name))

        image_ = hist_equalization(o)

        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_

    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image


def create_model(model_path, mi=512, ni=512, LOSS_FUNCTION='mse'):

    # TODO: change if using `channels_first` image data format
    input_img = Input(shape=(mi, ni, 1))

    # network definition
    c1e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1e)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2e = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2e)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3e = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3e)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4e = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4e)
    p4 = MaxPooling2D((2, 2), padding='same')(c4)

    c5e = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5e)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(c5)
    a4 = Concatenate(axis=3)([u4, c4])
    c6e = Conv2D(256, (3, 3), activation='relu', padding='same')(a4)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6e)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    a3 = Concatenate(axis=3)([u3, c3])
    c7e = Conv2D(128, (3, 3), activation='relu', padding='same')(a3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7e)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    a2 = Concatenate(axis=3)([u2, c2])
    c8e = Conv2D(64, (3, 3), activation='relu', padding='same')(a2)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8e)

    u1 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    a1 = Concatenate(axis=3)([u1, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(a1)

    c10 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    markers = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    cell_mask = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    output = Concatenate(axis=3)([markers, cell_mask])

    model = Model(input_img, output)
    model.compile(optimizer=Adam(lr=0.0001), loss=LOSS_FUNCTION)

    print ('Model was created')

    model.load_weights(model_path)

    return model


# postprocess markers
def postprocess_markers(img):

    # distance transform | only for circular objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    markers = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    new_m = ((hconvex(markers, 5) > 0) & (img > 240)).astype(np.uint8)
    
    # label connected components
    idx, res = cv2.connectedComponents(new_m)

    return idx, res


def hmax(img, h):
    
    h_img = img.astype(np.uint16) + h
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    rec0 = img

    # reconstruction
    for i in range(255):
        
        rec1 = np.minimum(cv2.dilate(rec0, kernel), h_img)
        if np.sum(rec0 - rec1) == 0:
            break
        rec0 = rec1
       
    # retype to uint8
    hmax_result = np.maximum(np.minimum((rec1 - h), 255), 0).astype(np.uint8)

    return hmax_result


def hconvex(img, h):
    return img - hmax(img, h)


# postprocess cell mask
def postprocess_cell_mask(b, threshold=229):

    # thresholding
    b = b.astype(np.uint8)
    bt = cv2.inRange(b, threshold, 255)

    return bt


def threshold_and_store(predictions, res_path):
    print(predictions.shape)

    for i in range(predictions.shape[0]):

        m = predictions[i, :, :, 1] * 255
        c = predictions[i, :, :, 3] * 255

        # postprocess the result of prediction
        idx, markers = postprocess_markers(m)
        cell_mask = postprocess_cell_mask(c)

        # correct border
        cell_mask = np.maximum(cell_mask, markers)

        labels = watershed(-c, markers, mask=cell_mask)
        labels = remove_edge_cells(labels)

        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(res_path, i), labels.astype(np.uint16))

def create_tracking(data_path, threshold=0.15):

    # check if the input path exists
    if not os.path.isdir(data_path):
        print('input path is not a valid path')
        return

    names = os.listdir(data_path)
    names = [name for name in names if '.tif' in name and 'mask' in name]
    names.sort()

    img = cv2.imread(os.path.join(data_path, names[0]), cv2.IMREAD_ANYDEPTH)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}


    old = np.zeros((mi, ni))
    index = 1
    n_images = len(names)

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(data_path, name), cv2.IMREAD_ANYDEPTH)

        labels = np.unique(img)[1:]

        parent_cells = []

        for label in labels:
            mask = (img == label) * 1

            mask_size = np.sum(mask)
            overlap = mask * old
            candidates = np.unique(overlap)[1:]

            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                score = np.sum(overlap == candidate * 1) / mask_size
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # no parent cell detected, create new track

                records[index] = [i, i, 0]
                result = result + mask * index
                index += 1
            else:

                if max_candidate not in parent_cells:
                    # prolonging track
                    records[max_candidate][1] = i
                    result = result + mask * max_candidate

                else:
                    # split operations
                    # if have not been done yet, modify original record
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        # find mask with max_candidate label in the result and rewrite it to index
                        m_mask = (result == max_candidate) * 1
                        result = result - m_mask * max_candidate + m_mask * index

                        records[index] = [i, i, max_candidate.astype(np.uint16)]
                        index += 1

                    # create new record with parent cell max_candidate
                    records[index] = [i, i, max_candidate.astype(np.uint16)]
                    result = result + mask * index
                    index += 1

                # update of used parent cells
                parent_cells.append(max_candidate)
        # store result
        cv2.imwrite(os.path.join(data_path, name), result.astype(np.uint16))
        old = result

    # store tracking
    print('Generating the tracking file.')
    with open(os.path.join(data_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write('{} {} {} {}\n'.format(key, records[key][0], records[key][1], records[key][2]))


def process_dataset(img_path, store_path, model_init_path):
    """
    reads images from the path and converts them to the np array
    """

    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    if not os.path.isfile(model_init_path):
        print('there is no model available')
        exit()

    if not os.path.isdir(img_path):
        print('the image datagiven name of dataset or the sequence is not valid')
        exit()

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    print(mi, ni)
    print(new_mi, new_ni)

    model = create_model(model_init_path, new_mi, new_ni)

    input_img = load_images(img_path, new_mi=new_mi, new_ni=new_ni)
                            
    pred_img = model.predict(input_img, batch_size=8)
    print('pred shape: {}'.format(pred_img.shape))

    pred_img = pred_img[:, :mi, :ni, :]

    threshold_and_store(pred_img, store_path)
    create_tracking(store_path)
