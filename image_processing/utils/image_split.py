import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')           # positional argument

    return parser.parse_args()

def main(args):
    img = cv2.imread(args.filename)
    cv2.imshow('whole image', img)

    h, w, channels = img.shape    
    half = w // 2
    left = img[:,:half]
    right = img[:,half:]
    cv2.imshow('left', left)
    cv2.imshow('right', right)

    left_filename = os.path.splitext(args.filename)[0] + "_0" + os.path.splitext(args.filename)[1]
    cv2.imwrite(left_filename, left)
    right_filename = os.path.splitext(args.filename)[0] + "_1" + os.path.splitext(args.filename)[1]
    cv2.imwrite(right_filename, right)

    cv2.waitKey(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)