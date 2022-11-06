import os
import argparse

import cv2

from background_remover import background_remover
from overlay_image import overlay_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_img', type=str, help='Path to the base image', required=True)
    parser.add_argument('--subject_img', type=str, help='Path to the subject image', required=True)
    parser.add_argument('--op_path', type=str, help='Output path to save', required=True)
    parser.add_argument('--remove_bg', type=bool, help='Enable background removal feature', required=False,
                        default=True)
    args = vars(parser.parse_args())
    base_image_path = args['base_img']
    subject_image_path = args['subject_img']
    # base_image_path = "C:/Users/keert/PycharmProjects/deep-painterly-harmonization/ip_data/background_image/background_base.jpg"
    # subject_image_path = "C:/Users/keert/PycharmProjects/deep-painterly-harmonization/ip_data/background_image/dog_sample_bg.jpg"
    subject_fg_path = os.path.join(args['op_path'],
                                   os.path.basename(subject_image_path).split(".")[0] + "_foreground." +
                                   os.path.basename(subject_image_path).split(".")[1])
    subject_mask_path = os.path.join(args['op_path'], os.path.basename(subject_image_path).split(".")[0] + "_mask." +
                                     os.path.basename(subject_image_path).split(".")[1])
    output_image_path = os.path.join(args['op_path'], os.path.basename(base_image_path).split(".")[0] + "_op." +
                                     os.path.basename(base_image_path).split(".")[1])
    base_image = cv2.imread(base_image_path)
    subject_image = cv2.imread(subject_image_path)
    subject_image_bgremoved, subject_image_bgremoved_mask = background_remover(image=subject_image)
    cv2.imwrite(subject_fg_path, subject_image_bgremoved)
    cv2.imwrite(subject_mask_path, subject_image_bgremoved_mask)
    output = overlay_image(base_image=base_image, subject_image=subject_fg_path, mask_image=subject_mask_path)
    cv2.imwrite(output_image_path, output)

