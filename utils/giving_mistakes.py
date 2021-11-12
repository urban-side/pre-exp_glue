import tensorflow_datasets as tfds


def make_miss_label(default_datasets, er_ratio, target_task):
    if er_ratio == 0:
        return default_datasets

    # タスクに応じて書き換え方が変わる
    if target_task == 'glue/cola':
        print('test')
