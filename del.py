import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import parse_annotation
import copy
import xml.etree.cElementTree as ET

def aug_image(train_instance, jitter=False):
    image_name = train_instance['filename']
    image = cv2.imread(image_name)

    if image is None: print('Cannot find ', image_name)

    h, w, c = image.shape
    all_objs = copy.deepcopy(train_instance['object'])

    if jitter:
        ### scale the image
        scale = np.random.uniform() / 10. + 1.
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        ### translate the image
        max_offx = (scale - 1.) * w
        max_offy = (scale - 1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        image = image[offy: (offy + h), offx: (offx + w)]

        ### flip the image
        flip = np.random.binomial(1, .5)
        if flip > 0.5: image = cv2.flip(image, 1)

        image = aug_pipe.augment_image(image)

        # resize the image to standard size
    image = cv2.resize(image, (1280, 1280))
    image = image[:, :, ::-1]

    # fix object's position and size
    for obj in all_objs:
        for attr in ['xmin', 'xmax']:
            if jitter: obj[attr] = int(obj[attr] * scale - offx)

            obj[attr] = int(obj[attr] * float(1280) / w)
            obj[attr] = max(min(obj[attr],1280), 0)

        for attr in ['ymin', 'ymax']:
            if jitter: obj[attr] = int(obj[attr] * scale - offy)

            obj[attr] = int(obj[attr] * float(1280) / h)
            obj[attr] = max(min(obj[attr], 1280), 0)

        if jitter and flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = 1280 - obj['xmax']
            obj['xmax'] = 1280 - xmin

    return image, all_objs

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug_pipe = iaa.Sequential(
            ["""
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )"""
            ],
            random_order=True
        )



ia.seed(1)
####################################################################################################################
def eslam(i,saved_name):
    train_imgs, train_labels = parse_annotation("/home/eslam/mcq_data/hard_c_ann/",
                                                "/home/eslam/mcq_data/hard_c/",
                                                ["taken-c", "not-c"])
    print('Seen labels:\t', train_labels)
    for train_instance in train_imgs[i:i+1]:
        image, all_objs = aug_image(train_instance, jitter=False)

    j = []
    for obj in all_objs:
        j.append( ia.BoundingBox(x1=obj['xmin'], y1=obj['ymin'], x2=obj['xmax']-1, y2=obj['ymax']) )

    bbs = ia.BoundingBoxesOnImage(j,shape=image.shape)
    print(bbs.bounding_boxes[0])
    #print(image1.shape)
    seq = iaa.Sequential([
        #iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            #translate_px={"x": 40, "y": 60},
            #scale=(0.95, 0.95)
            scale={"x": (0.90, 1.1), "y": (0.9, 1.1)},
            rotate=(-5,5),
            shear = (-10, 10)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, thickness=5)
    #image_after = bbs_aug.draw_on_image(image_aug, thickness=5, color=[0, 0, 255])
    image_after = bbs_aug.draw_on_image(image_aug, thickness=5)
    """
    plt.imshow(image_before)
    plt.show()
    plt.imshow(image_after)
    plt.show()
    """

    ######################################   Save the XML   #########################################
    root = ET.Element("annotation")
    size_element = ET.SubElement(root, "size")

    ET.SubElement(root, "filename").text = saved_name+".jpg"
    ET.SubElement(size_element, "width").text = "1280"
    ET.SubElement(size_element, "height").text = "1280"
    ET.SubElement(size_element, "depth").text = "3"

    for i in range(len(bbs.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
        name = all_objs[i]['name']
        object_element = ET.SubElement(root, "object")
        ET.SubElement(object_element, "name").text = name
        bndbox_element = ET.SubElement(object_element, "bndbox")
        ET.SubElement(bndbox_element, "xmin").text = str(after.x1)
        ET.SubElement(bndbox_element, "ymin").text = str(after.y1)
        ET.SubElement(bndbox_element, "xmax").text = str(after.x2)
        ET.SubElement(bndbox_element, "ymax").text = str(after.y2)

    tree = ET.ElementTree(root)
    tree.write("/home/eslam/"+saved_name+".xml")
    cv2.imwrite("/home/eslam/"+saved_name+".jpg",cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))



for i in range(10):
    saved = "269"+str(i)
    eslam(i,saved)

