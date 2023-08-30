import glob 
import os 
import shutil
import sys


def rename(path, name):
    jpgs = glob.glob(path + '/*.jpg')
    pngs = glob.glob(path + '/*.png')
    files = jpgs + pngs
    files.sort()
    for i, file in enumerate(files):
        shutil.move(file, path + '/' + name + str(i) + '.jpg')

argv = sys.argv

if len(argv) != 3:
    print('Usage: python rename.py [path] [name]')
    exit()

rename(argv[1], argv[2])
