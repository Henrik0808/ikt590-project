import os
import glob

subfolders = glob.glob("outputs/*/")

yes_no = None

while yes_no not in ('y', 'n'):
    yes_no = input("Are you sure you want to remove all files in the outputs folder (y/n)?\n")

    if yes_no == 'y':
        # Remove all files in the outputs folder
        for sf in subfolders:
            files = glob.glob(sf + '*')
            for f in files:
                os.remove(f)
        print('Deleted files in outputs folder')
        break
    elif yes_no == 'n':
        print('Aborted deletion script')
        break
    print('Please try again:\n')
