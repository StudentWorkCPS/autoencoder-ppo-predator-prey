import os 
import sys 
import glob
import gym_env.env_utils as env_utils
import shutil


def main():

    # Get the path of the folder containing the images
    if len(sys.argv) < 2:
        folder_path = 'results'
    else:
        folder_path = sys.argv[1]

    
    # Get the list of all the files in the folder
    path = os.path.join('imgs','*', '*.png')
    files = glob.glob(path)

    current_files = glob.glob(os.path.join(folder_path, '*.png'))
    current_count = len(current_files)
    
    print('Starting At:',current_count)
    for i,file in enumerate(files):
        print(i,file)
        shutil.move(file, '{}/img_{}.png'.format(folder_path, env_utils.number_to_n_digits(i+current_count,6)))

    print('Done! Total:',len(glob.glob(os.path.join(folder_path, '*.png'))))

if __name__ == '__main__':
    main()