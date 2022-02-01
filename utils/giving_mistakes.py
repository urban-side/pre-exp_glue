import random
import numpy as np


def make_miss_label(default_datasets, error_ratio, target_task):
    # tf.Tensor はそのままだと値をいじれないので numpy.array に変換
    temp_array = default_datasets.numpy()
    num_of_labels = len(temp_array)
    num_of_mistakes = int(num_of_labels * error_ratio)

    miss_info = f'We have {num_of_labels} labels: {error_ratio * 100}% error -> created {num_of_mistakes} mistakes\n'
    miss_info += f'Target task: {target_task} -> This task has {np.unique(temp_array)} label\n'

    # タスクに応じてラベルが変わるので、if分岐をかける
    # {0, 1} label ('glue/cola', 'glue/sst2', 'glue/mrpc', 'glue/qqp', 'glue/wnli')
    # {0: entailment, 1: not_entailment} label ('glue/qnli', 'glue/rte')
    zero_one_label_task = ['glue/cola', 'glue/sst2', 'glue/mrpc', 'glue/qqp', 'glue/qnli', 'glue/rte', 'glue/wnli']
    if target_task in zero_one_label_task:
        numzero = np.count_nonzero(temp_array == 0)
        numone = np.count_nonzero(temp_array == 1)

        sample_list = random.sample(range(num_of_labels), k=num_of_mistakes)
        for i in sample_list:
            temp_array[i] = 0 if temp_array[i] == 1 else 0

        processed_zero = np.count_nonzero(temp_array == 0)
        processed_one = np.count_nonzero(temp_array == 1)
        miss_info += f'Label details: [0: {numzero}, 1: {numone}] -> [0: {processed_zero}, 1: {processed_one}]'

    # {0: entailment, 1: contradiction, 2: neutral} label ('glue/mnli')
    elif target_task == 'glue/mnli':
        numzero = np.count_nonzero(temp_array == 0)
        numone = np.count_nonzero(temp_array == 1)
        numtwo = np.count_nonzero(temp_array == 2)
        zeroone = 0
        zerotwo = 0
        onezero = 0
        onetwo = 0
        twozero = 0
        twoone = 0

        sample_list = random.sample(range(num_of_labels), k=num_of_mistakes)
        for i in sample_list:
            if temp_array[i] == 0:
                temp_v = random.randint(1, 2)
                temp_array[i] = temp_v
                if temp_v == 1:  # 0 -> 1
                    zeroone += 1
                else:  # 0 -> 2
                    zerotwo += 1
            elif temp_array[i] == 1:
                temp_v + random.randint(0, 1)
                temp_array[i] = temp_v
                if temp_v == 0:  # 1 -> 0
                    onezero += 1
                else:  # 1 -> 2
                    onetwo += 1
            else:
                temp_v = random.randint(0, 1)
                temp_array[i] = temp_v
                if temp_v == 0:  # 2 -> 0
                    twozero += 1
                else:  # 2 -> 1
                    twoone += 1

        processed_zero = np.count_nonzero(temp_array == 0)
        processed_one = np.count_nonzero(temp_array == 1)
        processed_two = np.count_nonzero(temp_array == 2)
        miss_info += f'Error details:\n'
        miss_info += f'0 -> 1 = {zeroone}, 0 -> 2 = {zerotwo}, 1 -> 0 = {onezero}, 1 -> 2 = {onetwo}, 2 -> 0 = {twozero}, 2 -> 1 = {twoone}\n'
        miss_info += f'Label details: [0: {numzero}, 1: {numone}, 2: {numtwo}] -> [0: {processed_zero}, 1: {processed_one}, 2: {processed_two}]'

    # 最後に tf.Tensor に戻して返す
    return_tensor = tf.constant(temp_array)
    return return_tensor, miss_info