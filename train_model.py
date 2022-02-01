import os
import numpy as np
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa
from official.nlp import optimization
from utils.model_select import get_model_url
from utils.giving_mistakes import make_miss_label

tf.get_logger().setLevel('ERROR')

# 今回使うモデル選択
# 詳細：https://github.com/tensorflow/text/blob/master/docs/tutorials/bert_glue.ipynb
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tfhub_handle_encoder, tfhub_handle_preprocess = get_model_url(bert_model_name)

# glueのタスク選択（以下から選んでね）
# glue/cola, glue/sst2, glue/mrpc, glue/qqp, glue/mnli, glue/qnli, glue/rte, glue/wnli
tfds_name = 'glue/sst2'

# 学習パラメータ設定
epochs = 3
batch_size = 8
init_lr = 2e-5

# 各種pathの保存設定
datasets_path = './datasets'
main_save_path = './my_models'


def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

    Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
    """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


def load_dataset_from_tfds(in_memory_ds, info, split, batch_size, bert_preprocess_model, error_ratio=0):
    is_training = split.startswith('train')

    # エラーを付与する部分
    miss_info = ''
    if is_training and error_ratio > 0:
        in_memory_ds[split]['label'], miss_info = make_miss_label(in_memory_ds[split]['label'], error_ratio, tfds_name)
        print(miss_info)

    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()

    # To apply the preprocessing in all the inputs from the dataset, you will use the `map` function from the dataset.
    # The result is then cached for [performance](https://www.tensorflow.org/guide/data_performance#top_of_page).
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, num_examples, miss_info


def build_classifier_model(num_classes):
    """
    推論器の作成

    :param num_classes: 出力数の設定
    :return: 推論器のインスタンス
    """

    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model


def get_configuration(glue_task, num_train_steps, num_warmup_steps):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )

    if glue_task == 'glue/cola':
        metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

    return metrics, loss, optimizer


def save_bert_model(saved_model_dir, bert_preprocess_model, classifier_model, miss_info, error_ratio):
    preprocess_inputs = bert_preprocess_model.inputs
    bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
    bert_outputs = classifier_model(bert_encoder_inputs)
    model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

    # save error detail
    text_path = os.path.join(saved_model_dir, 'misstake_detail.txt')
    with open(text_path, 'a', encoding='UTF-8') as f:
        f.write(miss_info)

    # Save everything on the Colab host (even the variables from TPU memory)
    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    saved_model_path = os.path.join(saved_model_dir, f'ratio_{error_ratio}'.replace('.', ''))
    print('Saving', saved_model_path)
    model_for_export.save(saved_model_path, include_optimizer=False, options=save_options)


def print_example(which_target=''):
    if which_target == 'preprocess' or '':
        # 前処理用モデルのテスト
        test_preprocess_model = make_bert_preprocess_model(['my_input1', 'my_input2'])
        test_text = [np.array(['some random test sentence']), np.array(['another sentence'])]
        text_preprocessed = test_preprocess_model(test_text)
        print('Keys           : ', list(text_preprocessed.keys()))
        print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
        print('Word Ids       : ', text_preprocessed['input_word_ids'][0, :16])
        print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
        print('Input Mask     : ', text_preprocessed['input_mask'][0, :16])
        print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
        print('Type Ids       : ', text_preprocessed['input_type_ids'][0, :16])
        print()
    if which_target == 'classifier_model' or '':
        # 推論器のテスト
        test_classifier_model = build_classifier_model(2)
        bert_raw_result = test_classifier_model(text_preprocessed)
        print(tf.sigmoid(bert_raw_result))


def main():
    # 前処理モデルのロード
    bert_preprocess = hub.load(tfhub_handle_preprocess)

    # トークナイザのインスタンス化
    tok = bert_preprocess.tokenize(tf.constant(['Hello TensorFlow!']))
    print(tok, '\n')
    text_preprocessed = bert_preprocess.bert_pack_inputs([tok, tok], tf.constant(20))
    print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
    print('Word Ids       : ', text_preprocessed['input_word_ids'][0, :16])
    print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
    print('Input Mask     : ', text_preprocessed['input_mask'][0, :16])
    print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
    print('Type Ids       : ', text_preprocessed['input_type_ids'][0, :16])
    print()

    # 前処理と推論器のテスト表示
    # print_example()

    # モデル構造全体のイメージ画像を表示
    # tf.keras.utils.plot_model(test_preprocess_model, show_shapes=True, show_dtype=True)

    # 対象タスクのデータセットをロード
    tfds_info = tfds.builder(tfds_name, data_dir=datasets_path).info
    # pprint(tfds_info)

    # 必要部分のトリミング、ラベル種類等の確認
    sentence_features = list(tfds_info.features.keys())
    sentence_features.remove('idx')
    sentence_features.remove('label')
    available_splits = list(tfds_info.splits.keys())
    train_split = 'train'
    validation_split = 'validation'
    test_split = 'test'
    if tfds_name == 'glue/mnli':
        validation_split = 'validation_matched'
        test_split = 'test_matched'

    num_classes = tfds_info.features['label'].num_classes
    num_examples = tfds_info.splits.total_num_examples

    print(f"Using [{tfds_name}] from TFDS")
    print(f'This dataset has [{num_examples}] examples')
    print(f'Number of classes: {num_classes}')
    print(f'Features {sentence_features}')
    print(f'Splits {available_splits}')
    print()

    print(f'Fine tuning {tfhub_handle_encoder} model')
    bert_preprocess_model = make_bert_preprocess_model(sentence_features)

    # データセットのロードと整理
    in_memory_ds = tfds.load(tfds_name, data_dir=datasets_path, batch_size=-1, shuffle_files=True)

    # 複数GPUで処理
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        error_ratio_list = [0.0, 0.1, 0.2, 0.3]
        for error_ratio in error_ratio_list:
            train_dataset, train_data_size, miss_info = load_dataset_from_tfds(
                in_memory_ds,
                tfds_info,
                train_split,
                batch_size,
                bert_preprocess_model,
                error_ratio
            )
            validation_dataset, validation_data_size, _ = load_dataset_from_tfds(
                in_memory_ds,
                tfds_info,
                validation_split,
                batch_size,
                bert_preprocess_model
            )

            steps_per_epoch = train_data_size // batch_size
            num_train_steps = steps_per_epoch * epochs
            num_warmup_steps = num_train_steps // 10
            validation_steps = validation_data_size // batch_size

        # 推論器や誤差関数などの設定
        classifier_model = build_classifier_model(num_classes)
        metrics, loss, optimizer = get_configuration(tfds_name, num_train_steps, num_warmup_steps)
        classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

        # 学習の実行部分
        classifier_model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps
        )

            # モデルの保存
            bert_type = tfhub_handle_encoder.split('/')[-2]
            saved_model_name = f'{tfds_name.replace("/", "_")}_{bert_type}'
            saved_model_dir = os.path.join(main_save_path, saved_model_name)
            save_bert_model(saved_model_dir, bert_preprocess_model, classifier_model, miss_info, error_ratio)


if __name__ == '__main__':
    main()
