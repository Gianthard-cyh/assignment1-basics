import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from cs336_basics.tokenizer.tokenizer import Tokenizer

# 全局变量，用于在子进程中缓存 Tokenizer，避免每个分块都重新反序列化模型
_worker_tokenizer = None


def init_worker(model_path):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_file(model_path, ["<|endoftext|>"])


def encode_chunk(text_chunk):
    ids = _worker_tokenizer.encode(text_chunk)
    return np.array(ids, dtype=np.uint16)


def safe_file_chunker(filepath, chunk_bytes=10 * 1024 * 1024):
    """安全读取生成器：按块读取并自动对齐到下一个换行符或空格，防止截断 Token"""
    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break

            # 向后探读，直到遇到空白字符，确保词汇完整
            while True:
                char = f.read(1)
                if not char:
                    break
                chunk += char
                if char.isspace():
                    break

            # 优化：只 yield 文本，字节长度在外部计算，避免两次 next() 调用
            yield chunk


def main():
    parser = argparse.ArgumentParser(description="多进程文本 Tokenization")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="工作进程数")
    args = parser.parse_args()

    file_size = os.path.getsize(args.input_file)

    progress = Progress(
        TextColumn("[cyan]Tokenizing..."),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    # 预先分配列表存放带顺序索引的结果
    ordered_results = []

    with progress:
        task = progress.add_task("Processing", total=file_size)

        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.model_path,)) as executor:
            chunk_generator = safe_file_chunker(args.input_file)
            futures_to_order = {}
            order_idx = 0

            # 科学的“背压”控制：限制活跃任务数为核心数的 2 倍
            MAX_ACTIVE_TASKS = args.workers * 2

            while True:
                # 核心控制：如果任务队列满了，就阻塞等待至少一个任务完成
                if len(futures_to_order) >= MAX_ACTIVE_TASKS:
                    done, _ = wait(futures_to_order.keys(), return_when=FIRST_COMPLETED)
                    for f in done:
                        idx, byte_len = futures_to_order.pop(f)
                        ordered_results.append((idx, f.result()))
                        progress.update(task, advance=byte_len)

                try:
                    # 尝试读取下一块
                    text_chunk = next(chunk_generator)
                    # 精确计算 UTF-8 编码下的字节长度，用于进度条更新
                    chunk_byte_len = len(text_chunk.encode("utf-8"))

                    # 提交任务并记录顺序
                    future = executor.submit(encode_chunk, text_chunk)
                    futures_to_order[future] = (order_idx, chunk_byte_len)
                    order_idx += 1
                except StopIteration:
                    break

            # 处理最后剩下的“尾巴”任务
            for f in as_completed(futures_to_order):
                idx, byte_len = futures_to_order.pop(f)
                ordered_results.append((idx, f.result()))
                progress.update(task, advance=byte_len)

    print(f"所有进程处理完成，正在按顺序合并结果...")

    # 致命错误修复：必须按原始索引排序，否则多进程会导致文本乱序
    ordered_results.sort(key=lambda x: x[0])

    # 提取数组并高效拼接
    final_array = np.concatenate([res for _, res in ordered_results])
    np.save(args.output_file, final_array)
    
    # 打印统计信息
    compression_ratio = file_size / len(final_array)
    print(f"保存成功：{args.output_file}")
    print(f"总计 Tokens: {len(final_array)}")
    print(f"压缩比评估: {compression_ratio:.2f} chars/token")


if __name__ == "__main__":
    main()