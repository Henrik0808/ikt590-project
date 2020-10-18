import os

class cwd:
    def __init__(self, inner_path, outer_path=os.getcwd()):
        self.inner_path = inner_path
        self.outer_path = outer_path

    def __enter__(self):
        os.chdir(self.inner_path)

    def __exit__(self, *_):
        os.chdir(self.outer_path)
