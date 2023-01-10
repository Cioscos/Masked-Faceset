import os
from pathlib import Path
from io import BytesIO
import argparse
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

from xlib.DFLIMG.DFLJPG import DFLJPG
from xlib.interact import interact as io

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory: str):
    files = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                files.append(Path(path))

    return files


def process_image(file_info: tuple[Path, str]):
    input_dfl = DFLJPG.load(str(file_info[0]))

    if not input_dfl or not input_dfl.has_data():
        return None

    xseg_polys = None

    xseg = input_dfl.get_xseg_mask()

    if xseg is None:
        return None

    dfl_data = input_dfl.get_dict()
    landmarks = input_dfl.get_landmarks()
    if input_dfl.has_seg_ie_polys():
        xseg_polys = input_dfl.get_seg_ie_polys()
    face_type = input_dfl.get_face_type()

    img_a = input_dfl.get_img()

    xseg = cv2.resize(xseg, (img_a.shape[1], img_a.shape[0]))
    xseg = cv2.cvtColor(xseg, cv2.COLOR_GRAY2BGR)

    final = np.where(xseg, img_a, (0, 0, 0))

    _, buffer = cv2.imencode('.jpg', final)
    img_byte_arr = BytesIO(buffer)

    output_dfl_img = DFLJPG.load(file_info[1] + '/' + str(Path(file_info[0].name)),
                                 image_as_bytes=img_byte_arr.getvalue())
    output_dfl_img.set_dict(dfl_data)
    output_dfl_img.set_landmarks(landmarks)
    output_dfl_img.set_face_type(face_type)
    if xseg_polys:
        output_dfl_img.set_seg_ie_polys(xseg_polys)
    output_dfl_img.save()


def main():
    # manage input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, dest='src', required=True, help='Folder with source aligned images')
    parser.add_argument('-d', '--dst', type=str, dest='dst', help='Folder for result images', default='output')
    parser.add_argument('--cpus', type=int, dest='cpus', default=None, help='Number of cpus to use')

    args = parser.parse_args()

    dataset = make_dataset(args.src)

    # number of cpus to use
    cpus = args.cpus

    if cpus is None:
        cpus = io.input_int('Insert number of CPUs to use',
                            help_message='If the default option is selected it will use all cpu cores and it will slow down pc',
                            default_value=mp.cpu_count())

    # create output folder
    os.makedirs(args.dst, exist_ok=True)

    dataset = [(x, args.dst) for x in dataset]

    with mp.Pool(processes=cpus) as p:
        list(tqdm(p.imap_unordered(process_image, dataset),
                  desc=f"Creating masked dataset with {cpus} {'cpus' if cpus > 1 else 'cpu'}",
                  total=len(dataset), ascii=True))


if __name__ == '__main__':
    main()
