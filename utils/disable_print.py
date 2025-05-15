# -*- coding: utf-8 -*-
"""
禁用打印输出的工具模块
"""
import sys
import os
import builtins
import logging
import warnings

# 保存原始函数
_original_print = builtins.print
_original_open = builtins.open
_original_stdout = sys.stdout

class NullWriter:
    """空写入器，用于重定向标准输出"""
    def write(self, s): pass
    def flush(self): pass

def disable_print():
    """禁用所有打印输出"""
    # 禁用标准输出
    sys.stdout = NullWriter()
    
    # 禁用print函数
    builtins.print = lambda *args, **kwargs: None
    
    # 禁用日志
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    
    # 重定向open函数以防止创建作者相关文件
    def safe_open(*args, **kwargs):
        # 检查是否是试图创建作者相关文件
        if len(args) > 0 and isinstance(args[0], str):
            filename = args[0]
            if isinstance(filename, str) and (filename in ["使用须知.txt", "环境配置.txt"] or filename.endswith("使用须知.txt") or filename.endswith("环境配置.txt")):
                # 返回一个空的文件对象，但实际上不创建文件
                class DummyFile:
                    def write(self, x): pass
                    def read(self): return ""
                    def readline(self): return ""
                    def readlines(self): return []
                    def close(self): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def flush(self): pass
                return DummyFile()
        return _original_open(*args, **kwargs)
    
    builtins.open = safe_open
    
    # 忽略所有警告
    warnings.filterwarnings('ignore')

def enable_print():
    """恢复打印功能"""
    sys.stdout = _original_stdout
    builtins.print = _original_print
    builtins.open = _original_open

# 监控并删除作者相关文件
def monitor_and_delete_author_files():
    """定期检查并删除作者相关文件"""
    author_files = ["使用须知.txt", "环境配置.txt"]
    for file in author_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass

def disable_author_files():
    """禁用作者相关文件的创建，用于程序退出时"""
    author_files = ["使用须知.txt", "环境配置.txt"]

    # 劫持文件创建相关函数
    def no_op_open(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            path = args[0]
            if path in author_files or any(path.endswith(f) for f in author_files):
                from io import StringIO
                return StringIO()
        return _original_open(*args, **kwargs)
    
    # 替换内置函数
    builtins.open = no_op_open
    
    # 删除已存在的文件
    monitor_and_delete_author_files() 