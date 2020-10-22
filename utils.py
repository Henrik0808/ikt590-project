import os
import config


class cwd:
    def __init__(self, inner_path, outer_path=os.getcwd()):
        self.inner_path = inner_path
        self.outer_path = outer_path

    def __enter__(self):
        os.chdir(self.inner_path)

    def __exit__(self, *_):
        os.chdir(self.outer_path)


def get_categories():
    categories = config.categories

    categories_flat = []
    id2cat = {}
    cat2id = {}

    for boards in categories.values():
        for board in boards:
            categories_flat.append(board)

    # Needs to be alphabetically sorted,
    # because category_ids in make_dataset.py corresponds to target_names, which are sorted
    categories_flat = sorted(categories_flat)

    for category_id, (category, boards) in enumerate(categories.items()):
        cat2id[category] = category_id
        for board in boards:
            idx = categories_flat.index(board)
            id2cat[idx] = category

    return id2cat, cat2id, categories_flat