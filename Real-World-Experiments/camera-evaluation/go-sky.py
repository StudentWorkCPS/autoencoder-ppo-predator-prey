import cv2 
import matplotlib.pyplot as plt
import numpy as np
import calibration
import argparse

from fractions import Fraction
import datetime
import ffmpeg
import pandas as pd
import os

from imutils.video import FileVideoStream as FVS

def process_dir(dir):
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.MP4'):
            file = os.path.join(dir,file)
            process_file(file,headless=args.headless)

def process_file(file,headless=False):

    print('Processing file: ',file)

    meta_data = get_meta_data(file)

    fvs = FVS(file).start()

    paths = compute_paths(fvs,meta_data,args.headless)

    file = file.split('.')[0]
    paths.to_csv(file+'.csv',index=False)


def get_meta_data(file):
    media_file = file
    infos = ffmpeg.probe(media_file)["streams"]

    width = None
    height = None
    frame_rate = None
    duration = None
    created = None

    for info in infos:
        if info["codec_type"] == "video":
            width = int(info["width"])
            height = int(info["height"])
            frame_rate = float(Fraction(info["r_frame_rate"]))
            created = info["tags"]["creation_time"]
            duration = float(info["duration"])
            break
    time = datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%S.%fZ")
    created_dt = time.timestamp()

    return {'width':width,'height':height,'frame_rate':frame_rate,'duration':duration,'created':created, 'media_file':media_file,'created_dt':created_dt}
# to time.time() format



def step_to_time(step,meta_data):
    return meta_data['created_dt'] + step/meta_data['frame_rate']

def compute_paths(fvs,meta_data,headless=False):

    paths = pd.DataFrame(columns=['time','id','x','y'])
    step = 0
    
    while True:
        img = fvs.read() 

        if img is None:
            break

        step += 1
        if step % 3 != 0:
            continue

        img,center = calibration.detect_aruco(img,'center')
        found_ids = [x[0] for x in center]
        
        for (i,cx,cy,angle) in center:
            row = {'time':step_to_time(step,meta_data)}
            real_wolrd_pos = calibration.project(np.array([cx,cy]).astype(np.float64))

            img = cv2.putText(img, str(real_wolrd_pos),
                    (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            #if not hasattr(paths,str(i)):
            #    paths[str(i)] = [ for x in range(0,step)
            row['id'] = i
            row['x'] = real_wolrd_pos[0]
            row['y'] = real_wolrd_pos[1]
            row['angle'] = angle

            paths = pd.concat([paths,pd.DataFrame([row])])


        if not args.headless:
            cv2.imshow('TEST',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            if step % 100 == 0:
                print(step)

    return paths

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default='Iter0.MP4')
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--multi-dir', action='store_true', default=False)

    args = parser.parse_args()

    if args.multi_dir:

        dirs = args.file.split(',')
        print('Processing dirs: ',dirs)
        for d in dirs:
            print('Processing dir: ',d)
            process_dir(d)
    elif os.path.isdir(args.file):
        process_dir(args.file)
    else:
        process_file(args.file,headless=args.headless)
