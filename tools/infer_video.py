#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform a simple inference on a video file

Original source: https://github.com/cedrickchee/realtime-detectron/blob/master/inference.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
from core.test import im_detect_bbox
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis_video as vis_utils



c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_video)',
        default='/tmp/infer_video',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--detect_all',
        help='segmentation and keypoints',
        action='store_true'
    )
    parser.add_argument(
        'video_name',
        help='video filename',
        default=None,
        type=str
    )
    parser.add_argument(
        '--max_frames',
        default=None,
        type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()


    # TODO: draw only bboxes
    args.detect_all = True

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = 1983148141 # for some reason, the above doesn't work
    vid_writer = None


    cap = cv2.VideoCapture(args.video_name)
    i = 0

    while (cap.isOpened()):
        # Fetch image from camera
        _, im = cap.read()
        if vid_writer is None:
            print("input and output size is {}".format(im.shape[:2]))
            out_fn = os.path.join(args.output_dir, os.path.basename(args.video_name))
            print("writing to {}".format(out_fn))
            vid_writer = cv2.VideoWriter(out_fn, fourcc, 20.0, (im.shape[1], im.shape[0]))


        timers = defaultdict(Timer)
        t = time.time()

        try:
            with c2_utils.NamedCudaScope(0):
                if args.detect_all:
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, im, None, timers=timers)
                else:
                    raise NotImplementedError
                    scores, cls_boxes, scales = im_detect_bbox(model, im)
                    cls_segms = None
                    cls_keyps = None
    
            if i % 10 == 0:
                logger.info('processed %d frames', i)
                logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                for k, v in timers.items():
                    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    
            out_img = vis_utils.vis_one_image(
                        im[:, :, ::-1],  # BGR -> RGB for visualization
                        '',
                        args.output_dir,
                        cls_boxes,
                        cls_segms,
                        cls_keyps,
                        dataset=dummy_coco_dataset,
                        box_alpha=0.3,
                        show_class=args.detect_all,
                        thresh=0.7,
                        kp_thresh=2,
                        ext='jpg'  # default is PDF, but we want JPG.
                    )
        except AttributeError:
            print('Failed with frame {}'.format(i))
            raise

        assert out_img.shape == im.shape, 'in {} != out {}'.format(im.shape, out_img.shape)

        vid_writer.write(out_img)
        i += 1
        if args.max_frames is not None and i >= args.max_frames:
            print("Reached max frames ({})".format(i))

    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
