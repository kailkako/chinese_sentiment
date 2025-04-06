import os
import codecs

POS = os.path.join(os.getcwd(), 'pos')
NEG = os.path.join(os.getcwd(), 'neg')
FIX_POS = os.path.join(os.getcwd(), 'fix_pos')
FIX_NEG = os.path.join(os.getcwd(), 'fix_neg')


def fix_corpus(dir_s, dir_t):
    for item in os.listdir(dir_s):
        source_path = os.path.join(dir_s, item)
        target_path = os.path.join(dir_t, item)
        
        # 尝试用不同的编码读取文件
        try:
            # 尝试用 UTF-8 编码读取
            with open(source_path, 'r', encoding='utf-8') as f:
                s = f.read()
        except UnicodeDecodeError:
            try:
                # 如果 UTF-8 失败，尝试用 GBK 编码读取
                with open(source_path, 'r', encoding='gbk') as f:
                    s = f.read()
            except UnicodeDecodeError:
                try:
                    # 如果 GBK 失败，尝试用 GB2312 编码读取
                    with open(source_path, 'r', encoding='gb2312') as f:
                        s = f.read()
                except UnicodeDecodeError:
                    # 如果所有编码都失败，忽略错误
                    with open(source_path, 'r', encoding='gb2312', errors='ignore') as f:
                        s = f.read()
        
        # 将内容写入目标文件，使用 UTF-8 编码
        with codecs.open(target_path, 'w', encoding='utf-8') as ff:
            ff.write(s)


if __name__ == "__main__":
    # 确保目标目录存在
    if not os.path.isdir(FIX_POS):
        os.mkdir(FIX_POS)
    if not os.path.isdir(FIX_NEG):
        os.mkdir(FIX_NEG)
    
    # 处理正负样本
    fix_corpus(POS, FIX_POS)
    fix_corpus(NEG, FIX_NEG)