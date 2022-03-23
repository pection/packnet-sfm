# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())
import cv2
from glob import glob
from cv2 import imwrite
import matplotlib.pyplot as plt
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from PIL import Image
from numpy import asarray
import numpy as np
import cv2


def is_image(
    file,
    ext=(
        ".png",
        ".jpg",
    ),
):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PackNet-SfM inference of depth maps from images"
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, help="Input file or folder")
    parser.add_argument("--output", type=str, help="Output file or folder")
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs="+",
        default=None,
        help="Input and output image shape "
        "(default: checkpoint's config.datasets.augmentation.image_shape)",
    )
    parser.add_argument("--half", action="store_true", help="Use half precision (fp16)")
    parser.add_argument(
        "--save",
        type=str,
        choices=["npz", "png"],
        default=None,
        help="Save format (npz or png). Default is None (no depth map is saved).",
    )
    args = parser.parse_args()
    assert args.checkpoint.endswith(
        ".ckpt"
    ), "You need to provide a .ckpt file as checkpoint"
    assert (
        args.image_shape is None or len(args.image_shape) == 2
    ), "You need to provide a 2-dimensional tuple as shape (H,W)"
    assert (is_image(args.input) and is_image(args.output)) or (
        not is_image(args.input) and not is_image(args.input)
    ), "Input and output must both be images or folders"
    return args


@torch.no_grad()
def infer_and_save_depth(
    input_file, output_file, model_wrapper, image_shape, half, save
):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # print("IMAGE", type(image), image.shape, image.shape)
    image = resize_image(image, image_shape)
    print("IMAGE", type(image), image.size)
    image = to_tensor(image).unsqueeze(0)
    print("IMAGE", type(image), image.size, image.shape)

    # image2 = to_tensor(image)

    # print("IMAGE", type(image), image.shape, image.shape)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to("cuda:{}".format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)["inv_depths"][0]

    if save == "npz" or save == "png":
        # Get depth from predicted depth map and save to different formats
        filename = "{}.{}".format(os.path.splitext(output_file)[0], save)
        print(
            "Saving {} to {}".format(
                pcolor(input_file, "cyan", attrs=["bold"]),
                pcolor(filename, "magenta", attrs=["bold"]),
            )
        )
        # write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        viz_pred_inv_depth_cv2 = viz_pred_inv_depth[:, :, ::-1]
        # Concatenate both vertically
        print("viz_pred_inv_depth", viz_inv_depth(pred_inv_depth[0]))
        cv2.imwrite("test.png", viz_pred_inv_depth)
        cv2.imwrite(
            "/home/ai/work/data/media/images_resize/frame_rgb_1.png", rgb[:, :, ::-1]
        )
        cv2.imwrite(
            "/home/ai/work/data/media/images_resize/frame1_depth_1.png",
            viz_pred_inv_depth[:, :, ::-1],
        )
        cv2.imwrite(
            "/home/ai/work/data/media/images_resize/frame1_pred_1.png",
            viz_pred_inv_depth,
        )
        print("viz_pred_inv_depth", viz_pred_inv_depth)
        print("viz_pred_inv_depth", viz_pred_inv_depth.shape)
        print("TYPE ", type(viz_pred_inv_depth))
        # print(type(rgb))
        # print(viz_pred_inv_depth)
        # print(viz_pred_inv_depth[:, :, ::-1])
        # print(viz_pred_inv_depth.shape)
        # print(viz_pred_inv_depth[:, :, ::-1].shape)

        # print(type(viz_pred_inv_depth))
        img_n = convert_cv(viz_pred_inv_depth)
        img_r = convert_cv(rgb)
        # img = convert(viz_pred_inv_depth[:, :, ::-1], 0, 255, np.uint8)
        cv2.imshow("Window", img_n)
        cv2.imshow("Window2", img_r)

        image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print(
            "Saving {} to {}".format(
                pcolor(input_file, "cyan", attrs=["bold"]),
                pcolor(output_file, "magenta", attrs=["bold"]),
            )
        )
        # cv2.imwrite(output_file, image[:, :, ::-1])
        # print(viz_pred_inv_depth.shape)
        # image2 = Image.fromarray(viz_pred_inv_depth)
        # print(type(image2))

        # summarize image details
        # print(image2.size)
        # print(image2.mode)
        # cv2.imshow("image", image[:, :, ::-1])
        # cv2.imshow("rgb", rgb[:, :, ::-1])
        # cv2.imshow("inv depth", viz_pred_inv_depth)
        # plt.imshow(viz_pred_inv_depth[:, :, ::-1])
        # plt.show()
        # cv2.waitKey(0)


# img_n = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def convert_cv(img):
    return cv2.normalize(
        src=img,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )[:, :, ::-1]


def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to("cuda:{}".format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ["png", "jpg"]:
            files.extend(glob((os.path.join(args.input, "*.{}".format(ext)))))
        files.sort()
        print0("Found {} files".format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # Process each file
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow("frame", frame)
        cv2.imwrite("test.png", frame)
        infer_and_save_depth(
            "test.png", args.output, model_wrapper, image_shape, args.half, args.save
        )  # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # for fn in files[rank() :: world_size()]:
    #     infer_and_save_depth(
    #         fn, args.output, model_wrapper, image_shape, args.half, args.save
    #     )


if __name__ == "__main__":
    args = parse_args()
    main(args)
#
