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

# 输入函数
def parse_fn(line_words, line_tag):
    words = [w.encode() for w in line_words.strip().split()]
    tag = line_tag.strip().encode()
    return (words, len(words)), tag

def generator_fn(words_path, tags_path):
    with Path(words_path).open('r', encoding='utf-8') as f_words, Path(tags_path).open('r', encoding='utf-8') as f_tags:
        for line_words, line_tag in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tag)

def input_fn(words_path, tags_path, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}

    # 定义输出签名
    output_signature = (
        (
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(), dtype=tf.string)
    )

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words_path, tags_path),
        output_signature=output_signature
    )

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    # 批量处理和填充
    padded_shapes = (
        (
            tf.TensorShape([None]),
            tf.TensorShape([])
        ),
        tf.TensorShape([])
    )
    padding_values = (
        (
            b'<pad>',
            0
        ),
        b''
    )

    dataset = (dataset
               .padded_batch(
                   params.get('batch_size', 20),
                   padded_shapes=padded_shapes,
                   padding_values=padding_values
               )
               .prefetch(tf.data.AUTOTUNE))
    return dataset

# 模型定义
class LSTMClassifier(tf.keras.Model):
    def __init__(self, params):
        super(LSTMClassifier, self).__init__()
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
        self.tags_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.TextFileInitializer(
                params['tags'],
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER
            ),
            default_value=0
        )
        self.w2v = np.load(params['w2v'])['embeddings']
        self.w2v_var = tf.Variable(np.vstack([self.w2v, [[0.] * params['dim']]]), dtype=tf.float32, trainable=False)
        self.dropout_rate = params['dropout']
        self.lstm_size = params['lstm_size']
        self.num_tags = 0

        # 读取标签
        with Path(params['tags']).open('r', encoding='utf-8') as f:
            self.num_tags = sum(1 for _ in f)

        # LSTM 层
        self.lstm_fw = tf.keras.layers.LSTM(self.lstm_size, return_sequences=False)
        self.lstm_bw = tf.keras.layers.LSTM(self.lstm_size, return_sequences=False, go_backwards=True)
        self.bidirectional = tf.keras.layers.Bidirectional(self.lstm_fw, backward_layer=self.lstm_bw)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(self.num_tags)

    def call(self, inputs, training=False):
        words, nwords = inputs
        # 查找单词 ID
        word_ids = self.vocab_words.lookup(words)
        # 获取词向量
        embeddings = tf.nn.embedding_lookup(self.w2v_var, word_ids)
        embeddings = tf.keras.layers.Dropout(self.dropout_rate)(embeddings, training=training)

        # LSTM
        mask = tf.sequence_mask(nwords, maxlen=tf.shape(words)[1])
        lstm_output = self.bidirectional(embeddings, mask=mask)
        lstm_output = self.dropout(lstm_output, training=training)

        # 全连接层
        logits = self.dense(lstm_output)
        return logits

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            # 将标签从字符串转换为整数
            tags = self.tags_table.lookup(labels)
            loss = self.compiled_loss(tags, logits, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tags, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, labels = data
        logits = self(inputs, training=False)
        # 将标签从字符串转换为整数
        tags = self.tags_table.lookup(labels)
        self.compiled_loss(tags, logits, regularization_losses=self.losses)
        self.compiled_metrics.update_state(tags, logits)
        return {m.name: m.result() for m in self.metrics}

# 模型训练和评估
if __name__ == '__main__':
    params = {
        'dim': 300,
        'lstm_size': 32,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 5,
        'batch_size': 20,
        'buffer': 3500,
        'words': str(Path(DATA_DIR, 'vocab.words.txt')),
        'tags': str(Path(DATA_DIR, 'vocab.labels.txt')),
        'w2v': str(Path(DATA_DIR, 'w2v.npz'))
    }

    # 保存参数
    with Path('results/params.json').open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATA_DIR, f'{name}.words.txt'))

    def ftags(name):
        return str(Path(DATA_DIR, f'{name}.labels.txt'))

    # 创建模型
    model = LSTMClassifier(params)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 准备数据
    train_dataset = input_fn(fwords('train'), ftags('train'), params, shuffle_and_repeat=True)
    eval_dataset = input_fn(fwords('eval'), ftags('eval'), params)

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

        with Path(f'results/score/{name}.preds.txt').open('wb') as f:
            for golds, pred in zip(golds_gen, predictions):
                ((words, _), tag) = golds
                pred_label = params['tags'].splitlines()[np.argmax(pred)]
                f.write(b' '.join([tag, pred_label.encode(), b''.join(words)]) + b'\n')

    for name in ['train', 'eval']:
        write_predictions(name)