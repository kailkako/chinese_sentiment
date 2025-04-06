import sys
import json
import logging
import functools
import numpy as np
import tensorflow as tf
from pathlib import Path

# 配置日志
Path('results').mkdir(exist_ok=True)
tf.get_logger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('results/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = '../../data/hotel_comment'

# 创建标签映射字典
label_map = {}
with Path(DATA_DIR, 'vocab.labels.txt').open('r') as f:
    for idx, line in enumerate(f):
        label_map[line.strip()] = idx

# 输入函数
def parse_fn(line_words, line_tag):
    words = [w.strip() for w in line_words.strip().split()]
    tag = label_map.get(line_tag.strip(), 0)  # 使用标签映射，未知标签默认为0
    return words, tag

def generator_fn(words_path, tags_path):
    with Path(words_path).open('r') as f_words, Path(tags_path).open('r') as f_tags:
        for line_words, line_tag in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tag)

def input_fn(words_path, tags_path, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ([None], ())
    types = (tf.string, tf.int64)
    defaults = ('<pad>', tf.int64.min)  # 使用 int64 类型的最小值作为 padding 值

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words_path, tags_path),
        output_signature=(tf.TensorSpec(shape=shapes[0], dtype=tf.string),
                          tf.TensorSpec(shape=shapes[1], dtype=tf.int64))
    ).map(lambda w, t: (w[:params.get('nwords', 300)], t))

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(
                   params.get('batch_size', 20),
                   padded_shapes=([params.get('nwords', 300)], ()),
                   padding_values=defaults
               )
               .prefetch(tf.data.AUTOTUNE))
    return dataset

# 模型定义
class CNNClassifier(tf.keras.Model):
    def __init__(self, params):
        super(CNNClassifier, self).__init__()
        self.params = params
        self.vocab_words = tf.lookup.StaticHashTable(
            initializer=tf.lookup.TextFileInitializer(
                params['words'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ),
            default_value=params['num_oov_buckets']
        )
        self.w2v = np.load(params['w2v'])['embeddings']
        self.w2v_var = tf.Variable(np.vstack([self.w2v, [[0.] * params['dim']]]), dtype=tf.float32, trainable=False)
        self.filter_sizes = params['filter_sizes']
        self.num_filters = params['num_filters']
        self.dropout_rate = params['dropout']
        self.num_tags = len(label_map)  # 使用标签映射的长度

        # CNN 层
        self.convs = []
        for filter_size in self.filter_sizes:
            self.convs.append(
                tf.keras.layers.Conv2D(
                    filters=self.num_filters,
                    kernel_size=(filter_size, params['dim']),
                    activation='relu'
                )
            )
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(params['nwords'] - filter_size + 1, 1)
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(self.num_tags)

    def call(self, inputs, training=False):
        # 查找单词 ID
        word_ids = self.vocab_words.lookup(inputs)
        # 获取词向量
        embeddings = tf.nn.embedding_lookup(self.w2v_var, word_ids)
        embeddings = tf.expand_dims(embeddings, -1)

        # CNN
        pooled_outputs = []
        for conv in self.convs:
            conv_output = conv(embeddings)
            pooled = self.max_pool(conv_output)
            pooled_outputs.append(pooled)
        h_poll = tf.concat(pooled_outputs, axis=3)
        output = tf.reshape(h_poll, [-1, self.num_filters * len(self.filter_sizes)])

        # Dropout
        output = self.dropout(output, training=training)

        # 全连接层
        logits = self.dense(output)
        return logits

# 模型训练和评估
if __name__ == '__main__':
    params = {
        'dim': 300,
        'nwords': 300,
        'filter_sizes': [2, 3, 4],
        'num_filters': 64,
        'dropout': 0.6,
        'num_oov_buckets': 1,
        'epochs': 5,
        'batch_size': 20,
        'buffer': 3500,
        'words': str(Path(DATA_DIR, 'vocab.words.txt')),
        'tags': str(Path(DATA_DIR, 'vocab.labels.txt')),
        'w2v': str(Path(DATA_DIR, 'w2v.npz'))
    }

    # 保存参数
    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATA_DIR, f'{name}.words.txt'))

    def ftags(name):
        return str(Path(DATA_DIR, f'{name}.labels.txt'))

    # 创建模型
    model = CNNClassifier(params)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 准备数据
    train_dataset = input_fn(fwords('train'), ftags('train'), params, shuffle_and_repeat=True)
    eval_dataset = input_fn(fwords('eval'), ftags('eval'), params)

    # 打印数据集的结构和类型
    for data in train_dataset.take(1):
        print(data[0].shape, data[0].dtype)
        print(data[1].shape, data[1].dtype)

    # 训练和评估
    model.fit(
        train_dataset,
        epochs=params['epochs'],
        validation_data=eval_dataset
    )

    # 保存模型
    model.save('results/model')

    # 预测
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        test_dataset = input_fn(fwords(name), ftags(name), params)
        golds_gen = generator_fn(fwords(name), ftags(name))
        predictions = model.predict(test_dataset)

        with Path(f'results/score/{name}.preds.txt').open('w') as f:
            for golds, pred in zip(golds_gen, predictions):
                words, tag = golds
                pred_label = np.argmax(pred)
                f.write(' '.join([str(tag), str(pred_label), ' '.join(words)]) + '\n')

    for name in ['train', 'eval']:
        write_predictions(name)