import os


class cwd:
    def __init__(self, inner_path, outer_path=os.getcwd()):
        self.inner_path = inner_path
        self.outer_path = outer_path

    def __enter__(self):
        os.chdir(self.inner_path)

    def __exit__(self, *_):
        os.chdir(self.outer_path)


def get_categories():
    categories = {
        'sports': {'rec.sport.baseball', 'rec.sport.hockey'},
        'religion': {'alt.atheism', 'soc.religion.christian', 'talk.religion.misc'},
        'computers': {'comp.graphics', 'comp.os.ms-windows.misc'},
    }

    categories_flat = []
    id2cat = {}
    cat2id = {}

    idx = 0
    for category_id, (category, boards) in enumerate(categories.items()):
        cat2id[category] = category_id
        for board in boards:
            categories_flat.append(board)
            id2cat[idx] = category
            idx += 1

    return id2cat, cat2id, categories_flat
