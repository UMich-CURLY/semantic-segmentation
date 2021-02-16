"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

"""
Forest Dataset Loader
"""


import os
import os.path as path

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
import datasets.forest_labels as forest_labels
import datasets.uniform as uniform
from datasets.utils import make_dataset_folder

def forest_cv_split(root, split, cv_split):
    """
    90/10 train/val split, three random splits for cross validation

    split - train/val/test
    cv_split - 0,1,2,3

    cv_split == 3 means use train + val
    """
    img_dir_name = "training"
    img_path = os.path.join(root, img_dir_name, 'images')

    all_items = []
    c_items = os.listdir(img_path)
    c_items.sort()

    for it in c_items:
        all_items.append(os.path.join(img_path, it))

    if cv_split == 3:
        logx.msg('cv split {} {} {}'.format(split, cv_split, all_items))
        return all_items

    num_total_images = len(all_items)
    num_val_images = num_total_images // 10;

    offset = cv_split * num_total_images // cfg.DATASET.CV_SPLITS
    images = []
    for j in range(num_total_images):
        if j >= offset and j < (offset + num_val_images):
            if split == 'val':
                images.append(all_items[j])
        else:
            if split == 'train':
                images.append(all_items[j])

    logx.msg('cv split {} {} {}'.format(split, cv_split, images))
    return images

class Loader(BaseLoader):
    num_classes = 21
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='fine', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality, mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        ######################################################################
        # Forest-specific stuff:
        ######################################################################
        self.root = cfg.DATASET.FOREST_DIR
        self.id_to_trainid = forest_labels.label2trainid
        self.trainid_to_name = forest_labels.trainId2name
        self.fill_colormap()
        img_ext = 'png'
        mask_ext = 'png'
        img_dir_name = "training"

        img_root = path.join(self.root, img_dir_name, 'images')
        mask_root = path.join(self.root, img_dir_name, 'labels_id')
        
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            self.fine_images = forest_cv_split(self.root, mode, cfg.DATASET.CV)
            self.all_imgs = self.find_forest_images(
                self.fine_images, img_root, mask_root)

        logx.msg(f'cn num_classes {self.num_classes}')
        self.fine_centroids = uniform.build_centroids(self.all_imgs,
                                                      self.num_classes,
                                                      self.train,
                                                      cv=cfg.DATASET.CV,
                                                      id2trainid=self.id_to_trainid)
        self.centroids = self.fine_centroids

        self.build_epoch()

    def find_forest_images(self, images, img_root, mask_root):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        images: a list of images, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []

        for it in images:
            head, tail = os.path.split(it)
            item = (os.path.join(img_root, tail), os.path.join(mask_root, tail))
            items.append(item)

        logx.msg('mode {} found {} images'.format(self.mode, len(items)))

        return items

    def fill_colormap(self):
        palette = [29, 28, 33,
                   208, 235, 160,
                   43, 237, 21,
                   217, 240, 17,
                   186, 24, 65,
                   237, 9, 28,
                   235, 45, 98,
                   20, 99, 143,
                   157, 199, 194,
                   237, 61, 55,
                   32, 39, 232,
                   37, 193, 245,
                   132, 143, 127,
                   25, 151, 209,
                   83, 90, 169,
                   158, 163, 62,
                   182, 55, 127,
                   101, 28, 173,
                   162, 168, 104,
                   162, 135, 176,
                   45, 149, 238]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette
