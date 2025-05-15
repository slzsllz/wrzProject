# -*- coding: utf-8 -*-
"""
运行本项目需要python3.8及以下依赖库（完整库见requirements.txt）：
    opencv-python==4.5.5.64
    tensorflow==2.9.1
    PyQt5==5.15.6
    scikit-image==0.19.3
    torch==1.8.0
    keras==2.9.0
    Pillow==9.0.1
    scipy==1.8.0
点击运行主程序runMain.py，程序所在文件夹路径中请勿出现中文
"""

# 禁用所有打印输出 - 这必须在最开始导入，先于任何其他导入
from utils.disable_print import disable_print, enable_print, monitor_and_delete_author_files, disable_author_files
disable_print()  # 立即禁用所有打印

# 禁用内置print函数 - 双重保险
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None

# 设置标准输出为空设备 - 三重保险
import sys
import os

class NullWriter:
    def write(self, s): pass
    def flush(self): pass

_original_stdout = sys.stdout
sys.stdout = NullWriter()

import warnings
import logging

# 禁用所有日志输出
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# 禁用警告信息
warnings.filterwarnings('ignore')


author_files = ["使用须知.txt", "环境配置.txt"]
for file in author_files:
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception:
            pass

from GarbageRecing import Garbage_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
os.environ["QT_FONT_DPI"] = "150"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁止TensorFlow打印信息
warnings.filterwarnings(action='ignore')   # 忽略所有警告
os.environ["QT_SCALE_FACTOR"] = "1"  # 设置整体界面缩放为1.5倍

# 恢复标准输出，但仅用于错误报告
sys.stdout = _original_stdout
enable_print()
builtins.print = _original_print

if __name__ == '__main__':
    try:

        monitor_and_delete_author_files()
        
        # 注册退出清理函数
        import atexit
        
        def exit_handler():
            """程序退出前的清理工作"""

            monitor_and_delete_author_files()
            
            # 劫持文件操作函数以防止创建文件
            author_files = ["使用须知.txt", "环境配置.txt"]
            
            # 重定向内置的open函数
            def no_op_open(*args, **kwargs):
                if len(args) > 0 and isinstance(args[0], str):
                    filename = args[0]
                    if filename in author_files or filename.endswith("使用须知.txt") or filename.endswith("环境配置.txt"):
                        from io import StringIO
                        return StringIO()
                #return _original_open(*args, **kwargs)

            # 替换内置函数
            builtins.open = no_op_open
            
            # 再次检查并删除文件
            monitor_and_delete_author_files()
        
        # 注册退出处理函数
        atexit.register(exit_handler)
        
        app = QApplication(sys.argv)
        win = Garbage_MainWindow()
        
        # 启动时删除作者相关文件
        monitor_and_delete_author_files()
        
        # 设置Python解释器退出时的处理函数
        import signal
        def signal_handler(sig, frame):
            exit_handler()  # 调用之前定义的退出处理函数
            sys.exit(0)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        win.showTime()
        sys.exit(app.exec_())
    except Exception as e:
        # 确保在发生异常时也禁用作者文件
        disable_author_files()
        
        print(f"程序运行出错: {str(e)}")
        # 在控制台输出错误后退出
        import traceback
        traceback.print_exc()
