from rich import pretty
pretty.install()
from rich import traceback
traceback.install()


import regex


class ByteLevelBPETrainer:
    """字节级 BPE 训练器，支持特殊 token"""

    def __init__(self, special_tokens=None, pretokenize_pattern=None):
        self.merges = []  # 存储合并操作序列 (bytes, bytes)
        self.vocab = {}  # 词汇表: {id: bytes}
        self.special_tokens = special_tokens or []

        self.pretokenize_pattern = pretokenize_pattern or r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.compiled_pattern = regex.compile(self.pretokenize_pattern)

        # 构建特殊 token 模式，避免被切分
        if self.special_tokens:
            escaped_tokens = [regex.escape(token) for token in self.special_tokens]
            special_pattern = '|'.join(escaped_tokens)
            # 将特殊 token 模式和常规模式组合
            self.pretokenize_pattern = f"{special_pattern}|{self.pretokenize_pattern}"
            self.compiled_pattern = regex.compile(self.pretokenize_pattern)

    def _pretokenize(self, text):
        """预分词：使用正则表达式切分文本，保护特殊 token"""
        return self.compiled_pattern.findall(text)

    def train(self, text, num_merges, verbose=True):
        """
        训练 BPE 模型

        Args:
            text: 训练文本 (str 或 bytes)
            num_merges: 要执行的合并次数
            verbose: 是否打印训练过程
        """
        if isinstance(text, str):
            text = text.encode('utf-8')

        # 步骤1: 预分词并转换为字节序列
        text_str = text.decode('utf-8') if isinstance(text, bytes) else text
        pretokens = self._pretokenize(text_str)

        # 将预分词结果转换为字节序列
        word_tokens = []
        for token in pretokens:
            if token in self.special_tokens:
                # 特殊 token 作为整体，不可分割
                word_tokens.append([token.encode('utf-8')])
            else:
                # 常规 token 拆分为单字节
                token_bytes = token.encode('utf-8') if isinstance(token, str) else token
                word_tokens.append([bytes([b]) for b in token_bytes])

        if verbose:
            print(f"预分词完成: {len(pretokens)} 个片段")

        # 步骤2: 初始化词汇表，包含所有单字节
        for i in range(256):
            self.vocab[i] = bytes([i])

        # 将特殊 token 添加到词汇表
        next_id = 256
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            self.vocab[next_id] = token_bytes
            next_id += 1

        total_tokens = sum(len(word) for word in word_tokens)

        if verbose:
            print(f"初始 token 数量: {total_tokens}")
            print(f"初始词汇表大小: {len(self.vocab)}")
            print(f"特殊 token 数量: {len(self.special_tokens)}")
            print(f"\n开始训练，执行 {num_merges} 次合并...\n")

        # 步骤3: 执行指定次数的合并操作
        for i in range(num_merges):
            # 统计所有相邻 token 对的频率
            pair_counts = self._count_pairs_in_words(word_tokens)

            # 如果没有可合并的 token 对，提前停止
            if not pair_counts:
                if verbose:
                    print(f"警告: 没有更多可合并的 token 对，在第 {i} 次迭代后停止")
                break

            # 找到频率最高的 token 对
            best_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
            best_count = pair_counts[best_pair]

            if verbose:
                # 尝试将字节解码为可读字符串
                try:
                    token1_display = best_pair[0].decode('utf-8', errors='backslashreplace')
                    token2_display = best_pair[1].decode('utf-8', errors='backslashreplace')
                    merged_display = (best_pair[0] + best_pair[1]).decode('utf-8', errors='backslashreplace')
                except:
                    token1_display = repr(best_pair[0])
                    token2_display = repr(best_pair[1])
                    merged_display = repr(best_pair[0] + best_pair[1])

                print(f"合并 #{i+1}: '{token1_display}' + '{token2_display}' -> '{merged_display}' (频率: {best_count})")

            # 在所有单词中合并这个 token 对
            word_tokens = self._merge_pair_in_words(word_tokens, best_pair)

            # 记录合并操作
            self.merges.append(best_pair)

            # 将新合并的 token 加入词汇表
            new_token = best_pair[0] + best_pair[1]
            self.vocab[next_id] = new_token
            next_id += 1

            if verbose and (i + 1) % 10 == 0:
                total_tokens = sum(len(word) for word in word_tokens)
                print(f"  进度: {i+1}/{num_merges}, token 数: {total_tokens}, 词汇表大小: {len(self.vocab)}")

        if verbose:
            total_tokens = sum(len(word) for word in word_tokens)
            print(f"\n训练完成!")
            print(f"最终 token 数量: {total_tokens}")
            print(f"最终词汇表大小: {len(self.vocab)}")
            print(f"执行的合并次数: {len(self.merges)}")

        return word_tokens

    def _count_pairs_in_words(self, word_tokens):
        """统计所有单词中相邻 token 对的频率（不跨单词边界）"""
        pairs = {}
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_pair(self, tokens, pair):
        """在单个 token 列表中合并指定的 token 对"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            # 检查当前位置是否匹配要合并的 token 对
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2  # 跳过下一个 token
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _merge_pair_in_words(self, word_tokens, pair):
        """在所有单词中合并指定的 token 对"""
        return [self._merge_pair(word, pair) for word in word_tokens]

    def get_vocab(self):
        """返回词汇表 dict[int, bytes]"""
        return self.vocab

    def get_merges(self):
        """返回合并操作列表 list[tuple[bytes, bytes]]"""
        return self.merges


def adapters(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练字节级 BPE 分词器并返回词汇表和合并规则

    Args:
        input_path: 训练文本文件路径
        vocab_size: 目标词汇表大小 (必须 >= 256 + len(special_tokens))
        special_tokens: 需要保护的特殊 token 列表 (例如: ['<|endoftext|>'])

    Returns:
        vocab: token ID 到字节序列的映射字典
        merges: 合并操作列表，每项为 (bytes, bytes) 元组
    """
    # 验证词汇表大小
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size 必须至少为 {min_vocab_size} (256 基础字节 + {len(special_tokens)} 个特殊 token)")

    print(f"加载训练数据: {input_path}")

    # 读取训练文本
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"已加载 {len(text):,} 个字符\n")

    # 计算需要执行的合并次数
    num_merges = vocab_size - min_vocab_size

    print(f"目标词汇表大小: {vocab_size}")
    print(f"基础字节: 256")
    print(f"特殊 token: {len(special_tokens)}")
    print(f"需要执行的合并次数: {num_merges}\n")

    # 显示特殊 token
    if special_tokens:
        print("特殊 token:")
        for token in special_tokens:
            print(f"  {token}")
        print()

    # 训练 BPE 模型
    trainer = ByteLevelBPETrainer(special_tokens=special_tokens)
    trainer.train(text, num_merges=num_merges, verbose=True)

    # 返回词汇表和合并规则
    vocab = trainer.get_vocab()
    merges = trainer.get_merges()

    return vocab, merges


if __name__ == "__main__":
    # 简单演示
    text = "Hello world! This is a BPE trainer. <|endoftext|> It's working correctly."

    print("="*60)
    print("BPE 训练器演示")
    print("="*60)
    print(f"\n训练文本: {text}\n")

    trainer = ByteLevelBPETrainer(special_tokens=['<|endoftext|>'])
    trainer.train(text, num_merges=20, verbose=True)

    print("\n" + "="*60)
    print(f"学到的合并规则 (前 10 个):")
    print("="*60)
    for i, (a, b) in enumerate(trainer.get_merges()[:10], 1):
        try:
            a_str = a.decode('utf-8', errors='backslashreplace')
            b_str = b.decode('utf-8', errors='backslashreplace')
            merged_str = (a + b).decode('utf-8', errors='backslashreplace')
            print(f"{i:2d}. '{a_str}' + '{b_str}' = '{merged_str}'")
        except:
            print(f"{i:2d}. {repr(a)} + {repr(b)} = {repr(a+b)}")
