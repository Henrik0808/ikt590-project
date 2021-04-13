import os
import glob

# Get files and folders in the outputs folder
items = glob.glob("outputs/*")

yes_no = None

while yes_no not in ('y', 'n'):
    yes_no = input("Are you sure you want to remove all files in the outputs folder (y/n)?\n")

    if yes_no == 'y':
        # Remove all files in the outputs folder
        for i in items:
            # If i is folder, remove files in folder
            if os.path.isdir(i):
                files = glob.glob(i + '/*')
                for f in files:
                    os.remove(f)
            else:
                # Remove file
                os.remove(i)
        print('Deleted files in outputs folder')
        break
    elif yes_no == 'n':
        print('Aborted deletion script')
        break
    print('Please try again:\n')
