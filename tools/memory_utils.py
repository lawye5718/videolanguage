import torch
import gc
from loguru import logger

def clear_memory():
    """
    跨平台内存清理函数
    适用于Mac MPS、CUDA和CPU环境
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA显存已清理")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.info("MPS显存已清理")
    else:
        logger.info("CPU内存已清理")

def get_memory_info():
    """
    获取当前内存使用信息
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB"
    elif torch.backends.mps.is_available():
        # MPS没有直接的内存查询API，返回通用信息
        return "MPS - 使用统一内存架构"
    else:
        return "CPU - 使用系统内存"

# 用于测试
if __name__ == "__main__":
    print("当前内存状态:", get_memory_info())
    clear_memory()
    print("清理后内存状态:", get_memory_info())