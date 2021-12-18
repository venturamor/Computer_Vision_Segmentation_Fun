import pandas as pd
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# import skimage
from PIL import Image

# pytorch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def grabCut_on_images(dir_path):
    # help from here:
    # https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
    imgs_list = os.listdir(dir_path)
    N = 65
    iterations = 5
    bgdModel = np.zeros((1, N), np.float64)
    fgdModel = np.zeros((1, N), np.float64)ש
    # images names

    for img_ in imgs_list:
        # read image
        full_img_path = os.path.join(dir_path, img_)
        img = cv.imread(full_img_path, cv.COLOR_BGR2RGB)
        # display image
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(img_)
        plt.show()
        # grabCut prep
        mask = np.zeros(img.shape[:2], np.uint8)
        # gui select rect
        rect = cv.selectROI(img_, img)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img*mask2[:, :, np.newaxis]
        plt.imshow(img), plt.colorbar(), plt.show()
        # saving
        new_im_path = os.path.join(r'../output/' + 'segemnt_grabCut_' + img_)
        # TODO: uncomment when finish
        cv.imwrite(new_im_path, img)

def preprocess_deep(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def deep_segmentation(model, dir_path):
    # eval
    model.eval()
    imgs_list = os.listdir(dir_path)

    for img_ in imgs_list:
        full_img_path = os.path.join(dir_path, img_)
        input_image = Image.open(full_img_path)

        # create a mini-batch as expected by the model
        input_batch = preprocess_deep(input_image)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        img_new = np.array(r)[:, :, np.newaxis] * np.array(input_image)
        plt.imshow(img_new)
        plt.show()

        # saving
        new_im_path = os.path.join(r'../output/' + 'segemnt_deep_' + img_)
        cv.imwrite(new_im_path, img_new)


def forward_model_one_img(model_class, im_path):
    # for classification model
    input_image = Image.open(im_path)
    input_batch = preprocess_deep(input_image)
    output = model_class(input_batch)
    output_predictions = output.argmax(1).numpy()[0]
    return output_predictions



def paste_im_on_bkg_with_mask(img, bkg_img, mask_img):
    # padding img under assumption it smaller than background
    img_m = img * mask_img[:, :, np.newaxis]

    cols_delta = int((bkg_img.size[1] - img.size[1])/2)
    rows_delta = int((bkg_img.size[0] - img.size[0])/2)

    mask_padd = cv.copyMakeBorder(mask_img[:, :, np.newaxis], cols_delta, cols_delta, rows_delta, rows_delta, cv.BORDER_CONSTANT)
    bkg_m = bkg_img * (1 - mask_padd[:, :, np.newaxis])
    img_m_padd = cv.copyMakeBorder(img_m, cols_delta, cols_delta, rows_delta, rows_delta, cv.BORDER_CONSTANT)
    tot_bkg_img = img_m_padd + bkg_m

    return tot_bkg_img



if __name__ == '__main__':
    # Q1 - read the images from ./data/frogs and ./data/horses and display them

    frogs_path = r'./data/frogs'
    horses_path = r'./data/horses'

    # Q2 - picking classic method for segmentation - GrabCut
    grabCut_on_images(frogs_path)
    grabCut_on_images(horses_path)

    # Q2 - picking deep learn method for segmentation
    # Load a pre-trained

    # model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
    # model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    # model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)


    dir_path = frogs_path
    deep_segmentation(model, dir_path)

    dir_path = horses_path
    deep_segmentation(model, dir_path)

    # Q3 - 3 new images + Q4 - applying the method on them
    dir_path = r'./my_data/3_imgs'
    # Todo: uncomment when done
    grabCut_on_images(dir_path)
    deep_segmentation(model, dir_path)

    # Q6 - load classifier
    # We chose googlelenet
    #googlenet = torchvision.models.googlenet(pretrained=True)
    wide_resnet101_2 = torchvision.models.wide_resnet101_2(pretrained=True)
    model_class = wide_resnet101_2
    model_class.eval()

    # Q7
    # pathes
    cow_path = r'./data/cow.jpg'
    sheep_path = r'./data/sheep.jpg'
    im_path = sheep_path

    labels_path = r'./data/imagenet1000_clsidx_to_labels.txt'
    df_labels = pd.read_csv(labels_path, delimiter='\n')

    im_name = os.path.basename(im_path)

    # forward on the model
    output_predictions = forward_model_one_img(model_class, im_path)

    print('class for ' + im_name + df_labels.values[output_predictions])
    #["class for cow.jpg 217: 'English springer, English springer spaniel',"]

    # Q8 - segmenting one image that we classified
    sheep_path = r'./data/sheep'
    deep_segmentation(model, sheep_path)
    mask_sheep = grabCut_on_images(sheep_path)

    # Q9
    room_path = r'./data/room.jpg'
    bkg_img = Image.open(room_path)
    sheep_path = r'./data/sheep.jpg'
    sheep_img = Image.open(sheep_path)

    tot_sheep_bkg = paste_im_on_bkg_with_mask(sheep_img, bkg_img, mask_sheep)
    plt.imshow(tot_sheep_bkg)
    plt.axis('off')
    plt.show()

    new_im_name = 'sheep_in_room_.jpg'
    new_im_path = os.path.join(r'../output/', new_im_name)
    cv.imwrite(new_im_path, cv.cvtColor(tot_sheep_bkg, cv.COLOR_BGR2RGB))

    # Q10 - do Q7 again with the background illusion
    output_predictions = forward_model_one_img(model_class, new_im_path)
    print('class for ' + new_im_name + df_labels.values[output_predictions])
    # ["class for sheep_in_room_.jpg 832: 'stupa, tope',"]
    print('done')