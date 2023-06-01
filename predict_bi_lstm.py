"""
version 3.0 使用双向的lstm来预测，其中预测阶段的状态矩阵输入一共是粘贴好的4个
predict the results with the saved model in per batch
write it out
"""
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import data_generator
import define_models
import os


def check_file(file):
    if os.path.exists(file):
        print(file, " is exists!")
        os.remove(file)

max_input_len = 24 # 句子长度20，加标题21，再加四种tokens(<s>,</s>,<unk>)
max_target_len = 24
batch_size = 64
# 256维神经元
latent_dim = 256

def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Open the long_corpus  to generate the infer dataset in batch and predict the model.')
    parser.add_argument('--data-dir', type=str, help='The absolute path to long_corpus file.')
    parser.add_argument('--long-corpus', type=str, help='The name of long_corpus file.')
    parser.add_argument('--label-corpus', type=str, help='The name of label_corpus file.')
    parser.add_argument('--check-point-path', type=str, help='Path to save checkpoint file.')
    parser.add_argument('--model-hdf5-path', type=str, help='Path to save model.')
    parser.add_argument('--output', type=str, help='The predicted .fasta file.')
    args = parser.parse_args()
    # args: --data-dir /itmslhppc/itmsl0212/Projects/python/wrsCode/seq2seq/data/fan/
    # --long-corpus long_corpus.txt --check-point-path weight/cp.ckpt --model-hdf5-path weight/model.h5 --output predict.fasta

    return args

# 定义batch预测函数
def batch_decode_sequence(generator, num_batches):
    all_decoded_sentence_batch = []
    for batch_index in range(num_batches):
        # 从生成器中获取一批输入序列
        input_seq_batch = generator[batch_index][0]
        # 将输入序列编码为状态向量
        encoder_outputs, state_h, state_c = encoder_model.predict(input_seq_batch)
        # 生成一个形状为（batch_size，1）的空目标序列
        target_seq_batch = np.zeros((len(input_seq_batch), 1, target_vocab_size))
        # 将目标序列的第一个字符设置为开始字符
        target_seq_batch[:, 0, label_vocabulary['<s>']] = 1.

        # 循环生成目标序列的字符
        stop_condition = False
        decoded_sentence_batch = [''] * len(input_seq_batch)
        while not stop_condition:
            output_tokens_batch, h, c = decoder_model.predict(
                [target_seq_batch] + [state_h, state_c]) #分批次预测是为了分离梯度以减少计算量

            # 根据输出的概率值，采用argmax函数得到最有可能的字符索引
            sampled_token_index_batch = np.argmax(output_tokens_batch[:, -1, :], axis=-1)
            for i in range(len(input_seq_batch)):
                sampled_token_index = sampled_token_index_batch[i]
                sampled_char = label_reverse_vocabulary[sampled_token_index]
                decoded_sentence_batch[i] += sampled_char

                # 如果达到了序列的最大长度，或者生成了停止字符，停止生成字符
                if (sampled_char == '</s>'):
                    stop_condition = True

                # 更新目标序列，生成下一个字符
                target_seq_batch = np.zeros((len(input_seq_batch), 1, target_vocab_size))
                for i in range(len(input_seq_batch)):
                    target_seq_batch[i, 0, sampled_token_index_batch[i]] = 1.
                # 更新decoder的状态
                state_h, state_c = [h, c]

        all_decoded_sentence_batch += decoded_sentence_batch

    return all_decoded_sentence_batch


def write_batch_seq(output_file, all_decoded_sentence_batch):
    with open(output_file, mode='a') as f:
        for line in all_decoded_sentence_batch[0:len(input_texts)]:  # 填充在最末batch的东西我们直接不写出去了
            if line.startswith('>') == True or line.startswith(' ') == True:
                start = min(line.find('A'), line.find('T'), line.find('G'), line.find('C'))
                seq_name = line[0: start]  # 字符串名字含头不含尾
                seq_value = line[start:].rstrip('</s>').strip('<unk>').replace('*','')  # 去掉右边的截止字符
                f.write('\n')
                f.write(seq_name + '\n')
                f.write(seq_value)
            else:
                seq_value = line.rstrip('</s>').strip('<unk>').replace('*','')
                f.write(seq_value)

    f.close()
# 解析运行参数
parsed_args = build_parser()
data_dir = parsed_args.data_dir
long_corpus = data_dir + parsed_args.long_corpus
label_corpus = data_dir + parsed_args.label_corpus
check_point_path = data_dir + parsed_args.check_point_path
model_hdf5_path = data_dir + parsed_args.model_hdf5_path
output_file = data_dir + parsed_args.output
check_file(output_file)
# 打开制作好的词汇表文件，制作input_texts, target_texts, input_vocab_size, target_vocab_size, max_input_len, max_target_len, batch_size
# =====================================================================================
# Building long reads vocabulary
# Word string -> ID mapping
long_vocabulary = dict()
# Read the long_corpus file
long_vocabulary, long_lines_count = data_generator.read_corpus(long_corpus, long_vocabulary)
input_vocab_size = len(long_vocabulary)  # 输入端词汇字典的长度
# Build a reverse dictionary with the mapping ID -> word string
input_reverse_vocabulary = dict(zip(long_vocabulary.values(),long_vocabulary.keys()))
print("input_vocab_size:\t",input_vocab_size)

# Building labels vocabulary
# Contains label string -> ID mapping
label_vocabulary = dict()
# Read the long_corpus file
label_vocabulary, label_lines_count = data_generator.read_corpus(label_corpus, label_vocabulary)
target_vocab_size = len(label_vocabulary)  # 输出端词汇字典的长度
# Build a reverse dictionary with the mapping ID -> word string
# 反转输出序列的词汇表，以便将输出序列中的索引转换为单词
label_reverse_vocabulary = dict(zip(label_vocabulary.values(),label_vocabulary.keys()))
print("target_vocab_size:\t", target_vocab_size)

# 构建完词汇表以后从文件中逐行读并构建句子
# Read the source data file and read the lines,and append them as a list for token

input_texts = data_generator.append_line_texts(long_corpus, long_lines_count)
target_texts = data_generator.append_line_texts(label_corpus, label_lines_count)
# Make sure we extracted same number of both extracted source and target sentences



predict_generator = data_generator.predict_Seq2SeqDataGenerator(input_texts, target_texts
                                                , input_vocab_size, target_vocab_size
                                                , max_input_len, max_target_len, batch_size
                                                , long_vocabulary, label_vocabulary
                                                )
# 加载训练好的模型
model = load_model(model_hdf5_path)

_, encoder_model, decoder_model = define_models.bi_lstm_model(input_vocab_size, target_vocab_size, latent_dim)

encoder_model.load_weights(model_hdf5_path, by_name=True)
encoder_model.summary()
decoder_model.load_weights(model_hdf5_path, by_name=True)
decoder_model.summary()

all_decoded_sentence_batch = batch_decode_sequence(predict_generator, num_batches=len(input_texts)//batch_size + 1)  # len(input_texts)//batch_size+1


# with open(output_file, mode='a') as f:
#     for line in all_decoded_sentence_batch[0:len(input_texts)]:  # 填充在最末batch的东西我们直接不写出去了
#             f.write(line + '\n')
#测试阶段用上面，运行阶段用下面 先一行一行写出来

write_batch_seq(output_file, all_decoded_sentence_batch)

print("Predict success, the file is stored in:", output_file)

