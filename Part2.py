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
from skimage.segmentation import slic
from skimage.io import imread, imsave


from frame_video_convert import *
from Part1 import *

def deep_segmentation_part2(model, dir_path, output_path):
    # eval
    model.eval()
    imgs_list = os.listdir(dir_path)
    i = 0
    for img_ in imgs_list:
        i = i+1
        print('frame: ' + img_ +' number ' + str(i))
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

        mask = np.where((output_predictions.numpy() == 0), 0, 1).astype('uint8')

        img_new = input_image * mask[:, :, np.newaxis]
        # plt.imshow(img_new), plt.colorbar(), plt.show()

        # saving
        new_im_path = os.path.join(output_path , img_)
        cv.imwrite(new_im_path, cv.cvtColor(img_new, cv.COLOR_BGR2RGB))

def green_correct_segementation(segmented_path, output_path_fix):

    list_imgs_to_fix = os.listdir(segmented_path)
    for im_path in list_imgs_to_fix:
        full_im_path = os.path.join(segmented_path, im_path)
        img_to_fix = imread(full_im_path)
    # aa = slic(img_to_fix, start_label=0, n_segments=2)
    # seg_like = np.zeros_like(img_to_fix)
        G = img_to_fix[:, :, 1] # green channel
        bool = (G > 0) & (G < 140)
        ee = np.argwhere(bool)
        img_cpy = img_to_fix.copy()
        x = ee[:, 0]
        y = ee[:, 1]
        img_cpy[x, y, :] = 0
        plt.imshow(img_cpy), plt.show()
        # saving
        # new_im_path = os.path.join(output_path_fix, im_path)
        # cv.imwrite(new_im_path, cv.cvtColor(img_cpy, cv.COLOR_BGR2RGB))


def resize_and_replace(path, wanted_size):
    # resize to wanted_size = (cols, rows)
    list_imgs_to_fix = os.listdir(path)
    for img_ in list_imgs_to_fix:
        full_frame_path = os.path.join(path, img_)
        image = Image.open(full_frame_path)
        new_image = image.resize(wanted_size)     # (848, 480)
        new_image.save(full_frame_path)


def duplicate_frames(output_path_fix):
    import shutil
    list_imgs_to_fix = os.listdir(output_path_fix)
    # ['0000.jpg', '0053.jpg', '0084.jpg', '0145.jpg', '0188.jpg']
    wanted_final_frame = '0275.jpg'  # last frame in my vid
    list_imgs_to_fix.append(wanted_final_frame)
    N = len(list_imgs_to_fix)

    for i in range(N - 1):
        # frame num
        full_curr_frame_path = os.path.join(output_path_fix, list_imgs_to_fix[i])
        curr_frame_num = int(os.path.splitext(list_imgs_to_fix[i])[0])
        next_frame_num = int(os.path.splitext(list_imgs_to_fix[i + 1])[0])
        # copy frame until next one
        for j in range(curr_frame_num + 1, next_frame_num):
            new_frame_str = str(j).zfill(4) + '.jpg'
            full_new_frame_path = os.path.join(output_path_fix, new_frame_str)
            shutil.copyfile(full_curr_frame_path, full_new_frame_path)


def combine_frames_bkg(me_dir, shot_dir, bkg_path, new_frames_path):

    bkg_img = imread(bkg_path)
    me_frames = os.listdir(me_dir)
    shot_frames = os.listdir(shot_dir)
    N = len(me_frames)

    for i in range(N):
        me_frame_path = os.path.join(me_dir, me_frames[i])
        shot_frame_path = os.path.join(shot_dir, shot_frames[i])

        new_frame = bkg_img.copy()
        shot_frame = imread(shot_frame_path)
        me_frame = imread(me_frame_path)

        # mask me
        mask_me = me_frame[:, :, 1] > 30
        ee = np.argwhere(mask_me)
        x_me = ee[:, 0]
        y_me = ee[:, 1]

        # apply me on bkg
        new_frame[x_me, y_me, :] = me_frame[x_me, y_me, :]
        # plt.imshow(new_frame), plt.show()

        # mask shot
        mask_shot = shot_frame[:, :, 1] > 30
        ee = np.argwhere(mask_shot)
        x_shot = ee[:, 0]
        y_shot = ee[:, 1]

        # apply shot on bkg
        new_frame[x_shot, y_shot, :] = shot_frame[x_shot, y_shot, :]
        # plt.imshow(new_frame), plt.show()

        # saving
        new_frame_path = os.path.join(new_frames_path, shot_frames[i])
        cv.imwrite(new_frame_path, cv.cvtColor(new_frame, cv.COLOR_BGR2RGB))


if __name__ == '__main__':
    # Q1

    # C:\Users\ventu\Documents\zoom\2021-05-27 17.26.09 mor.ventura@campus.technion.ac.il's zoom meeting 94383878715

    vid_path = r'./my_data/shooting_data/My_vid_shooting.mp4'
    output_path = r'../output/vid_shooting_frames'

    # video_to_image_seq(vid_path, output_path=output_path)

    # Q2
    #frames_list = os.listdir(output_path)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    segmented_path = r'../output/seg_me_shhot'

    deep_segmentation_part2(model, output_path, segmented_path)

    count_frames = len(os.listdir(segmented_path))

    # Q3
    vid_3_path = r'./my_data/shooting_data/Bullet_shots.mp4'
    output3_path = r'../output/shorten_shot_frames'

    video_to_image_seq(vid_3_path, output_path=output3_path)

    segmented_path_3 = r'../output/shorten_seg'

    # deep_segmentation_part2(model, output3_path, segmented_path_3) - bas results
    grabCut_on_images(output3_path)

    # green correction over segmented images
    output_path_fix = r'../output/shorten_seg_fixed'
    green_correct_segementation(segmented_path_3, output_path_fix)

    # resize
    resize_and_replace(output_path_fix, (848, 480))

    # duplicating relevant frames
    duplicate_frames(output_path_fix)

    # Q4

    bkg_dir = r'./my_data/background'
    resize_and_replace(bkg_dir, (848, 480))

    # background
    full_bkg_path = r'./my_data/background/westworld.jpg'

    # new frames
    new_frames_path = r'../output/new_frames_video'

    # combine
    me_dir = r'../output/seg_me_shhot'
    combine_frames_bkg(me_dir, output_path_fix, full_bkg_path, new_frames_path)

    # frames to vid
    image_seq_to_video(new_frames_path, output_path='./my_data/final_vid.mp4', fps=15.0)

    print('done')

