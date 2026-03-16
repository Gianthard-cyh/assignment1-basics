import argparse
import numpy as np
from cs336_basics.tokenizer.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="预览编码后的二进制数据并检查非法 ID")
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="npy文件路径")
    parser.add_argument("--n", type=int, default=100, help="预览前n个Token")
    parser.add_argument("--check_all", action="store_true", help="是否扫描全文件检查非法 ID")
    args = parser.parse_args()

    # 1. 加载 Tokenizer 并获取词表大小
    tokenizer = Tokenizer.from_file(args.model_path)
    vocab_size = len(tokenizer.vocab)

    # 2. 内存映射加载
    data = np.load(args.input_file, mmap_mode="r")

    # 3. 预览逻辑
    sample_ids = data[0 : args.n].tolist()
    decoded_text = tokenizer.decode(sample_ids)
    print(f"预览前 {args.n} 个 Tokens (Vocab Size: {vocab_size})")
    print(f"原始 ID 序列: {sample_ids}")
    print(f"解码内容:\n{decoded_text}\n" + "-" * 30)

    # 4. 新增：全量扫描非法 ID
    if args.check_all:
        print("正在扫描全文件以检查超出词表的 ID...")
        # 利用 numpy 的矢量化操作，mmap 会按需分块读取磁盘，速度极快
        out_of_bounds_mask = (data < 0) | (data >= vocab_size)
        
        if np.any(out_of_bounds_mask):
            invalid_indices = np.where(out_of_bounds_mask)[0]
            invalid_values = data[invalid_indices]
            print(f"❌ 发现非法 ID！数量: {len(invalid_indices)}")
            # 打印前 10 个异常位置示例
            for idx, val in zip(invalid_indices[:10], invalid_values[:10]):
                print(f"位置 {idx}: ID 为 {val}")
        else:
            print("✅ 检查完毕：所有 ID 均在词表范围内。")


if __name__ == "__main__":
    main()