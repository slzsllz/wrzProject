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

# 禁用所有打印输出 - 这里我们不直接导入，因为runMain.py已经导入了
# 从而避免重复禁用造成的问题
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: None

# 禁用所有日志输出
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# 导入所需包
import argparse
import os
import random
import time
from os import getcwd
import locale
import sys

import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont, QBrush, QPen, QColor, QPainter, QImage, QPixmap
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

from Garbage.label_name import Chinese_name
from models.experimental import attempt_load
from utils.__init__ import QMainWindow
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device, time_synchronized

# 添加字体文件检查函数
def check_fonts():
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # Windows黑体
        "C:/Windows/Fonts/simsun.ttc",  # Windows宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux文泉驿微米黑
        "/System/Library/Fonts/PingFang.ttc"  # macOS苹方
    ]
    
    available_fonts = []
    for path in font_paths:
        if os.path.exists(path):
            available_fonts.append(path)
    
    if available_fonts:
        return available_fonts[0]
    else:
        return None

# 添加推理线程类
class InferenceThread(QThread):
    # 定义信号
    inference_finished = pyqtSignal(object, object, float)
    progress_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, img, device, half, opt):
        super().__init__()
        self.model = model
        self.img = img
        self.device = device
        self.half = half
        self.opt = opt

    def run(self):
        try:
            self.progress_updated.emit("正在处理...")
            
            t1 = time_synchronized()
            pred = self.model(self.img, augment=False)[0]
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, 
                                      classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()
            
            inferTime = round((t2 - t1), 2)
            self.inference_finished.emit(pred, self.img, inferTime)
            
        except Exception as e:
            import traceback
            print(f"推理过程出错: {str(e)}")
            traceback.print_exc()
            self.error_occurred.emit(f"处理出错: {str(e)}")


class Garbage_MainWindow(QMainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(Garbage_MainWindow, self).__init__(*args, **kwargs)
        # 移除 author_flag
        # self.author_flag = False  # 已删除

        # 设置文件监控定时器
        self.file_monitor_timer = QtCore.QTimer()
        self.file_monitor_timer.timeout.connect(self.check_and_delete_author_files)
        self.file_monitor_timer.start(100)  # 每100毫秒检查一次
        
        # 初始化时删除作者相关文件
        self.check_and_delete_author_files()

        self.setupUi(self)  # 界面生成
        self.retranslateUi(self)  # 界面控件
        self.setUiStyle(window_flag=True, transBack_flag=True)  # 设置界面样式

        self.path = getcwd()
        self.video_path = getcwd()

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.timer_video = QtCore.QTimer()  # 视频定时器
        self.flag_timer = ""  # 用于标记正在进行的功能项（视频/摄像）

        self.LoadModel()  # 加载预训练模型
        self.slot_init()  # 定义槽函数
        self.files = []  #
        self.cap_video = None  # 视频流对象
        self.CAM_NUM = 0  # 摄像头标号
        self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象

        self.detInfo = []
        self.current_image = []
        self.detected_image = None
        # self.dataset = None
        self.count = 0  # 表格行数，用于记录识别识别条目
        self.res_set = []  # 用于历史结果记录的列表
        self.c_video = 0
        self.count_name = ["可降解", "纸板", "玻璃", "金属", "纸质", "塑料"]
        self.count_table = []
        self.plotBar(self.count_name, [0 for i in self.count_name], self.colors, margin=30)

        # 检查中文字体
        self.chinese_font_path = check_fonts()

        # 添加表格初始化设置
        self.init_table()

    def init_table(self):
        """初始化表格设置"""
        try:
            # 设置表格列数和列标题
            self.tableWidget.setColumnCount(5)
            self.tableWidget.setHorizontalHeaderLabels(['序号', '文件路径', '类别', '位置', '置信度'])
            
            # 设置表格列宽
            self.tableWidget.setColumnWidth(0, 60)    # 序号列
            self.tableWidget.setColumnWidth(1, 200)   # 文件路径列
            self.tableWidget.setColumnWidth(2, 100)   # 类别列
            self.tableWidget.setColumnWidth(3, 150)   # 位置列
            self.tableWidget.setColumnWidth(4, 80)    # 置信度列
            
            # 清空表格内容，确保从第一行开始添加
            self.tableWidget.setRowCount(0)
            
            # 设置表格样式
            self.tableWidget.setStyleSheet("""
                QTableWidget {
                    background-color: rgba(0, 180, 200, 80);
                    color: white;
                    gridline-color: rgba(255, 255, 255, 60);
                    alternate-background-color: rgba(0, 170, 190, 100);
                    border: none;
                }
                QTableWidget::item {
                    padding: 5px;
                    border-bottom: 1px solid rgba(255, 255, 255, 30);
                }
                QTableWidget::item:selected {
                    background-color: rgba(0, 200, 255, 150);
                }
                QHeaderView::section {
                    background-color: rgba(0, 150, 180, 150);
                    color: white;
                    padding: 5px;
                    border: none;
                    border-bottom: 2px solid rgba(255, 255, 255, 40);
                    font-weight: bold;
                }
                QTableWidget QTableCornerButton::section {
                    background-color: rgba(0, 150, 180, 150);
                    border: none;
                }
                QTableWidget QScrollBar {
                    background-color: rgba(0, 180, 200, 60);
                }
                QTableWidget QScrollBar::handle {
                    background-color: rgba(0, 220, 255, 150);
                }
            """)
            
            # 设置表格选择模式
            self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # 整行选择
            self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)  # 单行选择
            
            # 设置表格自动调整特性
            self.tableWidget.horizontalHeader().setStretchLastSection(True)  # 最后一列自动拉伸
            self.tableWidget.verticalHeader().setVisible(False)  # 隐藏垂直表头
            
            # 启用交替行颜色
            self.tableWidget.setAlternatingRowColors(True)
            
            # 设置表格网格线
            self.tableWidget.setShowGrid(True)
            
        except Exception as e:
            print(f"初始化表格出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def slot_init(self):
        self.toolButton_file.clicked.connect(self.choose_file)
        self.toolButton_folder.clicked.connect(self.choose_folder)
        self.toolButton_video.clicked.connect(self.button_open_video_click)
        self.timer_video.timeout.connect(self.show_video)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_model.clicked.connect(self.choose_model)
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        self.tableWidget.cellPressed.connect(self.table_review)
        self.toolButton_saveing.clicked.connect(self.save_file)

    def table_review(self, row, col):
        try:
            if col == 0:  # 点击第一列时
                this_path = self.tableWidget.item(row, 1)  # 表格中的文件路径
                res = self.tableWidget.item(row, 2)  # 表格中记录的识别结果
                axes = self.tableWidget.item(row, 3)  # 表格中记录的坐标

                if (this_path is not None) & (res is not None) & (axes is not None):
                    this_path = this_path.text()
                    if os.path.exists(this_path):
                        res = res.text()
                        axes = axes.text()

                        image = self.cv_imread(this_path)  # 读取选择的图片
                        image = cv2.resize(image, (850, 500))

                        axes = [int(i) for i in axes.split(",")]
                        confi = float(self.tableWidget.item(row, 4).text())

                        # print(axes)
                        # image = self.drawRectBox(image, axes, res)
                        count = self.count_table[row]
                        self.plotBar(self.count_name, count, self.colors, margin=30)
                        self.label_numer_result.setText(str(sum(count)))
                        image = self.drawRectEdge(image, axes, alpha=0.2, addText=res)
                        # 在Qt界面中显示检测完成画面
                        self.display_image(image)  # 在界面中显示画面

                        # 在界面标签中显示结果
                        self.label_xmin_result.setText(str(int(axes[0])))
                        self.label_ymin_result.setText(str(int(axes[1])))
                        self.label_xmax_result.setText(str(int(axes[2])))
                        self.label_ymax_result.setText(str(int(axes[3])))
                        self.label_score_result.setText(str(round(confi * 100, 2)) + "%")
                        self.label_class_result.setText(res)

                        QtWidgets.QApplication.processEvents()
        except:
            self.label_display.setText('重现表格记录时出错，请检查表格内容！')
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")

    def LoadModel(self, model_path=None):
        """
        读取预训练模型
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./weights/garbage-best.pt',
                            help='model.pt path(s)')  # 模型路径仅支持.pt文件
        parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')  # 检测图像大小，仅支持480
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 置信度阈值
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # NMS阈值
        # 选中运行机器的GPU或者cpu，有GPU则GPU，没有则cpu，若想仅使用cpu，可以填cpu即可
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-dir', type=str, default='inference', help='directory to save results')  # 文件保存路径
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')  # 分开类别
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 使用NMS
        self.opt = parser.parse_args()  # opt局部变量，重要
        out, weight, imgsz = self.opt.save_dir, self.opt.weights, self.opt.img_size  # 得到文件保存路径，文件权重路径，图像尺寸
        self.device = select_device(self.opt.device)  # 检验计算单元,gpu还是cpu
        self.half = self.device.type != 'cpu'  # 如果使用gpu则进行半精度推理
        if model_path:
            weight = model_path
        self.model = attempt_load(weight, map_location=self.device)  # 读取模型
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # 检查图像尺寸
        if self.half:  # 如果是半精度推理
            self.model.half()  # 转换模型的格式
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 得到模型训练的类别名
        # self.names = [Chinese_name[i] for i in self.names]
        for i, v in enumerate(self.names):
            if v in Chinese_name.keys():
                self.names[i] = Chinese_name[v]
        # hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
        #        '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        color = [[132, 56, 255], [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
                 [72, 249, 10], [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187],
                 [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49],
                 [44, 153, 168], [0, 194, 255], [52, 69, 147], [100, 115, 255], [0, 24, 236]]
        self.colors = color if len(self.names) <= len(color) else [[random.randint(0, 255) for _ in range(3)] for _ in
                                                                   range(len(self.names))]  # 给每个类别一个颜色
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # 创建一个图像进行预推理
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # 预推理

    def choose_model(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        self.comboBox_select.clear()  # 下拉选框的显示
        self.comboBox_select.addItem('所有目标')  # 清除下拉选框
        self.clearUI()  # 清除UI上的label显示
        self.flag_timer = ""
        # 调用文件选择对话框
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "选取图片文件", getcwd(),  # 起始路径
                                                                "Model File (*.pt)")  # 文件类型
        # 显示提示信息
        if fileName_choose != '':
            self.toolButton_model.setToolTip(fileName_choose + ' 已选中')
        else:
            fileName_choose = None  # 模型默认路径
            self.toolButton_model.setToolTip('使用默认模型')
        self.LoadModel(fileName_choose)

    def select_obj(self):
        QtWidgets.QApplication.processEvents()
        if self.flag_timer == "video":
            # 打开定时器
            self.timer_video.start(30)
        elif self.flag_timer == "camera":
            self.timer_camera.start(30)

        ind = self.comboBox_select.currentIndex() - 1
        ind_select = ind
        if ind <= -1:
            ind_select = 0
        # else:
        #     ind_select = len(self.detInfo) - ind - 1
        if len(self.detInfo) > 0:
            # self.label_class_result.setFont(font)
            self.label_class_result.setText(self.detInfo[ind_select][0])  # 显示类别
            self.label_score_result.setText(str(self.detInfo[ind_select][2]))  # 显示置信度值
            # 显示位置坐标
            self.label_xmin_result.setText(str(int(self.detInfo[ind_select][1][0])))
            self.label_ymin_result.setText(str(int(self.detInfo[ind_select][1][1])))
            self.label_xmax_result.setText(str(int(self.detInfo[ind_select][1][2])))
            self.label_ymax_result.setText(str(int(self.detInfo[ind_select][1][3])))

        image = self.current_image.copy()
        if len(self.detInfo) > 0:
            for i, box in enumerate(self.detInfo):  # 遍历所有标记框
                if ind != -1:
                    if ind != i:
                        continue
                # 在图像上标记目标框

                label = '%s %.0f%%' % (box[0], float(box[2]) * 100)
                self.label_score_result.setText(box[2])
                # label = str(box[0]) + " " + str(float(box[2])*100)
                # 画出检测到的目标物
                # self.names. box[0]
                image = self.drawRectBox(image, box[1], addText=label, color=self.colors[box[3]])

            # self.label_score_result.setText(str(len(self.detInfo) - count))
            # 在Qt界面中显示检测完成画面
            self.display_image(image)
            # self.label_display.display_image(image)

    def choose_folder(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        self.c_video = 0
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        self.comboBox_select.clear()  # 下拉选框的显示
        self.comboBox_select.addItem('所有目标')  # 清除下拉选框
        self.clearUI()  # 清除UI上的label显示

        self.flag_timer = ""
        # 选择文件夹
        dir_choose = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", self.path)
        self.path = dir_choose  # 保存路径
        if dir_choose != "":
            self.textEdit_pic.setText(dir_choose + '文件夹已选中')
            self.label_display.setText('正在启动识别系统...\n\nleading')
            QtWidgets.QApplication.processEvents()

            rootdir = os.path.join(self.path)
            for (dirpath, dirnames, filenames) in os.walk(rootdir):
                for filename in filenames:
                    temp_type = os.path.splitext(filename)[1]
                    if temp_type == '.png' or temp_type == '.jpg' or temp_type == '.jpeg':
                        img_path = dirpath + '/' + filename
                        image = self.cv_imread(img_path)  # 读取选择的图片
                        image = cv2.resize(image, (850, 500))
                        img0 = image.copy()
                        img = letterbox(img0, new_shape=self.imgsz)[0]
                        img = np.stack(img, 0)
                        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        img = np.ascontiguousarray(img)

                        img = torch.from_numpy(img).to(self.device)  # 把图像矩阵移至到训练单元中(GPU中或CPU中)
                        img = img.half() if self.half else img.float()  # 如果是半精度则转换图像格式
                        img /= 255.0  # 归一化
                        if img.ndimension() == 3:  # 如果图像时三维的添加1维变成4维
                            img = img.unsqueeze(0)
                        t1 = time_synchronized()  # 推理开始时间
                        pred = self.model(img, augment=False)[0]  # 前向推理
                        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                                   classes=self.opt.classes,
                                                   agnostic=self.opt.agnostic_nms)  # NMS过滤
                        t2 = time_synchronized()  # 结束时间
                        det = pred[0]

                        p, s, im0 = None, '', img0
                        self.current_image = img0.copy()
                        # save_path = str(Path(self.opt.save_dir) / Path(p).name)  # 文件保存路径
                        if det is not None and len(det):  # 如果有检测信息则进入
                            # self.label_numer_result.setText(str(len(det)))  # 将检测个数放置到主界面中
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
                            number_i = 0  # 类别预编号
                            self.detInfo = []

                            count = [0 for i in self.count_name]
                            for *xyxy, conf, cls in reversed(det):  # 遍历检测信息
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                # 将检测信息添加到字典中
                                self.detInfo.append(
                                    [self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                                number_i += 1  # 编号数+1

                                class_name = self.names[int(cls)]
                                
                                # 设置UI标签
                                self.label_class_result.setText(class_name)
                                
                                # 确保UI字体支持中文
                                from PyQt5.QtGui import QFont
                                font = QFont("SimHei", 12)  # 使用黑体
                                self.label_class_result.setFont(font)

                                for cn in range(len(self.count_name)):
                                    if self.names[int(cls)] == self.count_name[cn]:
                                        count[cn] += 1

                                self.label_score_result.setText('%.2f' % conf)
                                label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)

                                # 画出检测到的目标物
                                # print(xyxy)
                                im0 = self.drawRectBox(im0, xyxy, alpha=0.2, addText=label, color=self.colors[int(cls)])
                                self.label_xmin_result.setText(str(c1[0]))
                                self.label_ymin_result.setText(str(c1[1]))
                                self.label_xmax_result.setText(str(c2[0]))
                                self.label_ymax_result.setText(str(c2[1]))

                                # 将结果记录至列表中
                                res_all = [self.names[int(cls)], conf.item(), [c1[0], c1[1], c2[0], c2[1]]]
                                self.res_set.append(res_all)
                                self.change_table(img_path, res_all[0], res_all[2], res_all[1])

                            for _ in range(len(det)):
                                self.count_table.append(count)  # 记录各个类别数目
                            self.plotBar(self.count_name, count, self.colors, margin=30)
                            self.label_numer_result.setText(str(sum(count)))
                            # self.label_score_result.setText(str(len(det) - count))
                            # 更新下拉选框
                            self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                            self.comboBox_select.clear()
                            self.comboBox_select.addItem('所有目标')
                            for i in range(len(self.detInfo)):
                                text = "{}-{}".format(self.detInfo[i][0], i + 1)
                                self.comboBox_select.addItem(text)
                            self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                            image = im0.copy()
                            InferenceNms = t2 - t1  # 单张图片推理时间
                            self.label_time_result.setText(str(round(InferenceNms, 2)))  # 将推理时间放到右上角

                        else:
                            # 清除UI上的label显示
                            self.label_numer_result.setText("0")
                            self.label_class_result.setText('0')
                            # font = QtGui.QFont()
                            # font.setPointSize(16)
                            # self.label_class_result.setFont(font)
                            self.label_score_result.setText("0")  # 显示置信度值
                            # 清除位置坐标
                            self.label_xmin_result.setText("0")
                            self.label_ymin_result.setText("0")
                            self.label_xmax_result.setText("0")
                            self.label_ymax_result.setText("0")

                        # 在Qt界面中显示检测完成画面
                        self.detected_image = image.copy()
                        self.display_image(image)  # 在界面中显示画面
                        QtWidgets.QApplication.processEvents()
                        # self.label_display.display_image(image)

        else:
            self.clearUI()

    def choose_file(self):
        """图像检测"""
        try:
            self.timer_camera.stop()
            self.timer_video.stop()
            self.c_video = 0
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.cap_video:
                self.cap_video.release()  # 释放视频画面帧

            self.comboBox_select.clear()  # 下拉选框的显示
            self.comboBox_select.addItem('所有目标')  # 清除下拉选框
            self.clearUI()  # 清除UI上的label显示

            self.flag_timer = ""
            # 使用文件选择对话框选择图片
            fileName_choose, filetype = QFileDialog.getOpenFileName(
                self.centralwidget, "选取图片文件",
                self.path,  # 起始路径
                "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
            self.path = fileName_choose  # 保存路径

            if fileName_choose != '':
                self.flag_timer = "image"
                self.textEdit_pic.setText(fileName_choose + '文件已选中')
                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:
                    image = self.cv_imread(self.path)  # 读取选择的图片
                    if image is None:
                        self.label_display.setText("无法读取图片文件，请检查文件格式")
                        return
                        
                    image = cv2.resize(image, (850, 500))
                    img0 = image.copy()
                    img = letterbox(img0, new_shape=self.imgsz)[0]
                    img = np.stack(img, 0)
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                    img = np.ascontiguousarray(img)

                    # 转换为张量
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    
                    # 使用线程处理推理
                    self.current_image = img0.copy()  # 保存原始图像
                    self.inference_thread = InferenceThread(self.model, img, self.device, self.half, self.opt)
                    self.inference_thread.inference_finished.connect(self.process_image_result)
                    self.inference_thread.progress_updated.connect(self.update_progress)
                    self.inference_thread.error_occurred.connect(self.handle_error)
                    self.inference_thread.start()
                    
                except Exception as e:
                    print(f"处理图片时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.label_display.setText(f"处理图片时出错: {str(e)}")
                    
            else:
                self.clearUI()
        except Exception as e:
            print(f"选择文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def process_image_result(self, pred, img, inferTime):
        try:
            det = pred[0]
            im0 = self.current_image.copy()
            
            self.label_time_result.setText(str(inferTime))
            
            if det is not None and len(det):
                # 缩放坐标到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                number_i = 0
                self.detInfo = []
                count = [0 for i in self.count_name]
                
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    self.detInfo.append([self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1
                    
                    class_name = self.names[int(cls)]
                    
                    # 设置UI标签
                    self.label_class_result.setText(class_name)
                    
                    # 确保UI字体支持中文
                    from PyQt5.QtGui import QFont
                    font = QFont("SimHei", 12)  # 使用黑体
                    self.label_class_result.setFont(font)

                    for cn in range(len(self.count_name)):
                        if self.names[int(cls)] == self.count_name[cn]:
                            count[cn] += 1
                            
                    self.label_score_result.setText('%.2f' % conf)
                    label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)
                    
                    # 画框
                    im0 = self.drawRectBox(im0, xyxy, alpha=0.2, addText=label, color=self.colors[int(cls)])
                    
                    # 更新坐标显示
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))
                    
                    # 记录结果
                    res_all = [self.names[int(cls)], conf.item(), [c1[0], c1[1], c2[0], c2[1]]]
                    self.res_set.append(res_all)
                    self.change_table(self.path, res_all[0], res_all[2], res_all[1])
                    
                for _ in range(len(det)):
                    self.count_table.append(count)
                    
                self.label_numer_result.setText(str(sum(count)))
                self.plotBar(self.count_name, count, self.colors, margin=30)
                
                # 更新下拉框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    text = "{}-{}".format(self.detInfo[i][0], i + 1)
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                
            else:
                # 无检测结果
                self.label_numer_result.setText("0")
                self.label_class_result.setText('0')
                self.label_score_result.setText("0")
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                
            # 显示结果图像
            self.detected_image = im0.copy()
            self.display_image(im0)
            
            # 删除作者相关文件
            author_files = ["使用须知.txt", "环境配置.txt"]
            for file in author_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except Exception as e:
                        pass
            
        except Exception as e:
            print(f"处理推理结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.label_display.setText(f"处理结果时出错: {str(e)}")

    def update_progress(self, message):
        self.label_display.setText(message)
        QtWidgets.QApplication.processEvents()
        
    def handle_error(self, message):
        self.label_display.setText(message)
        QtWidgets.QApplication.processEvents()

    def button_open_video_click(self):
        self.c_video = 0
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.cap:
            self.cap.release()

        self.clearUI()  # 清除显示
        QtWidgets.QApplication.processEvents()

        if not self.timer_video.isActive():  # 检查定时状态
            # 弹出文件选择框选择视频文件
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,  # 起始路径
                                                                    "视频(*.mp4;*.avi)")  # 文件类型
            self.video_path = fileName_choose

            if fileName_choose != '':
                self.flag_timer = "video"
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.setStyleText(self.textEdit_video)

                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:  # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                except:
                    print("[INFO] could not determine # of frames in video")

                self.timer_video.start(30)  # 打开定时器

            else:
                # 选择取消，恢复界面状态
                self.flag_timer = ""
                self.clearUI()

        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_video.stop()
            self.cap_video.release()
            self.label_display.clear()
            time.sleep(0.5)
            self.clearUI()
            self.comboBox_select.clear()
            self.comboBox_select.addItem('所有目标')
            QtWidgets.QApplication.processEvents()

    def show_video(self):
        try:
            # 定时器槽函数，每隔一段时间执行
            flag, image = self.cap_video.read()  # 获取画面

            if flag:
                image = cv2.resize(image, (850, 500))
                self.current_image = image.copy()

                img0 = image.copy()
                img = letterbox(img0, new_shape=self.imgsz)[0]
                img = np.stack(img, 0)
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                # 转换为张量
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                    
                # 使用线程处理推理
                self.inference_thread = InferenceThread(self.model, img, self.device, self.half, self.opt)
                self.inference_thread.inference_finished.connect(self.process_video_result)
                self.inference_thread.error_occurred.connect(self.handle_error)
                self.inference_thread.start()
                
                # 暂停定时器，等待推理完成
                self.timer_video.stop()
                
            else:
                self.timer_video.stop()
                if self.cap_video:
                    self.cap_video.release()
                
        except Exception as e:
            print(f"视频处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.timer_video.stop()
            if self.cap_video:
                self.cap_video.release()
            self.label_display.setText(f"视频处理出错: {str(e)}")

    def process_video_result(self, pred, img, inferTime):
        try:
            det = pred[0]
            im0 = self.current_image.copy()
            
            self.label_time_result.setText(str(inferTime))
            
            if det is not None and len(det):
                # 缩放坐标到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                number_i = 0
                self.detInfo = []
                count = [0 for i in self.count_name]
                
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    self.detInfo.append([self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1
                    
                    class_name = self.names[int(cls)]
                    
                    # 设置UI标签
                    self.label_class_result.setText(class_name)
                    
                    # 确保UI字体支持中文
                    from PyQt5.QtGui import QFont
                    font = QFont("SimHei", 12)  # 使用黑体
                    self.label_class_result.setFont(font)

                    for cn in range(len(self.count_name)):
                        if self.names[int(cls)] == self.count_name[cn]:
                            count[cn] += 1
                            
                    self.label_score_result.setText('%.2f' % conf)
                    label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)
                    
                    # 画框
                    im0 = self.drawRectBox(im0, xyxy, alpha=0.2, addText=label, color=self.colors[int(cls)])
                    
                    # 更新坐标显示
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))
                    
                    # 记录结果 - 和camera模式类似，每10帧记录一次
                    self.c_video += 1
                    if self.c_video % 10 == 0:
                        res_all = [self.names[int(cls)], conf.item(), [c1[0], c1[1], c2[0], c2[1]]]
                        self.res_set.append(res_all)
                        self.change_table(str(self.count), res_all[0], res_all[2], res_all[1])
                
                # 更新柱状图
                self.plotBar(self.count_name, count, self.colors, margin=30)
                self.label_numer_result.setText(str(sum(count)))
                
                # 更新下拉框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    text = "{}-{}".format(self.detInfo[i][0], i + 1)
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                
            else:
                # 无检测结果
                self.label_numer_result.setText("0")
                self.label_class_result.setText('0')
                self.label_score_result.setText("0")
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                
            # 显示结果图像
            self.detected_image = im0.copy()
            self.display_image(im0)
            
            # 重新启动定时器获取下一帧
            self.timer_video.start(30)
            
        except Exception as e:
            print(f"处理视频结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.timer_video.stop()
            if self.cap_video:
                self.cap_video.release()
            self.label_display.setText(f"处理视频结果时出错: {str(e)}")

    def button_open_camera_click(self):
        self.c_video = 0
        if self.timer_video.isActive():
            self.timer_video.stop()
        # self.timer_camera.stop()
        QtWidgets.QApplication.processEvents()

        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                              u"请检测相机与电脑是否连接正确！ ",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                # 准备运行识别程序
                self.flag_timer = "camera"
                self.clearUI()

                self.textEdit_camera.setText('实时摄像已启动')
                self.setStyleText(self.textEdit_camera)
                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                self.timer_camera.start(30)  # 打开定时器
        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.clearUI()
            QtWidgets.QApplication.processEvents()

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        # if self.flag:
        flag, image = self.cap.read()  # 获取画面
        if flag:
            self.current_image = image.copy()

            s = np.stack([letterbox(x, new_shape=self.imgsz)[0].shape for x in [image]], 0)
            rect = np.unique(s, axis=0).shape[0] == 1
            img0 = [image].copy()
            img = [letterbox(x, new_shape=self.imgsz, auto=rect)[0] for x in img0]
            img = np.stack(img, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)

            pred, useTime = self.predict(img)
            self.label_time_result.setText(str(useTime))

            det = pred[0]
            p, s, im0 = None, '', img0
            count = [0 for i in self.count_name]
            if det is not None and len(det):  # 如果有检测信息则进入
                # self.label_numer_result.setText(str(len(det)))  # 将检测个数放置到主界面中

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0[0].shape).round()  # 把图像缩放至im0的尺寸

                number_i = 0  # 类别预编号
                self.detInfo = []
                for *xyxy, conf, cls in reversed(det):  # 遍历检测信息
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # 将检测信息添加到字典中
                    self.detInfo.append([self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1  # 编号数+1

                    # image = im0[0].copy()
                    class_name = self.names[int(cls)]
                    self.label_class_result.setText(class_name)
                    
                    # 确保UI字体支持中文
                    from PyQt5.QtGui import QFont
                    font = QFont("SimHei", 12)  # 使用黑体
                    self.label_class_result.setFont(font)

                    self.label_score_result.setText('%.2f' % conf)
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    for cn in range(len(self.count_name)):
                        if self.names[int(cls)] == self.count_name[cn]:
                            count[cn] += 1
                    label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)
                    # 画出检测到的目标物
                    image = self.drawRectBox(image, xyxy, addText=label, color=self.colors[int(cls)])
                    self.label_xmin_result.setText(str(c1[0]))
                    self.label_ymin_result.setText(str(c1[1]))
                    self.label_xmax_result.setText(str(c2[0]))
                    self.label_ymax_result.setText(str(c2[1]))

                    # 将结果记录至列表中
                    self.c_video += 1
                    if self.c_video % 10 == 0:
                        res_all = [self.names[int(cls)], conf.item(), [c1[0], c1[1], c2[0], c2[1]]]
                        self.res_set.append(res_all)
                        self.change_table(str(self.count), res_all[0], res_all[2], res_all[1])
                        self.count_table.append(count)  # 记录各个类别数目

                # self.label_score_result.setText(str(len(det) - count))
                self.plotBar(self.count_name, count, self.colors, margin=30)
                self.label_numer_result.setText(str(sum(count)))

                # 更新下拉选框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.detInfo)):
                    text = "{}-{}".format(self.detInfo[i][0], i + 1)
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

            else:
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                # self.label_time_result.setText('0 s')
                self.label_class_result.setText('0')
                # font = QtGui.QFont()
                # font.setPointSize(16)
                # self.label_class_result.setFont(font)
                self.label_score_result.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

            self.detected_image = image.copy()
            # 在Qt界面中显示检测完成画面
            self.display_image(image)
            # self.label_display.display_image(image)
        else:
            self.timer_video.stop()

    def predict(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        t2 = time_synchronized()
        InferNms = round((t2 - t1), 2)

        return pred, InferNms

    def save_file(self):
        if self.detected_image is not None:
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            cv2.imwrite('./pic_' + str(now_time) + '.png', self.detected_image)
            QMessageBox.about(self.centralwidget, "保存文件", "\nSuccessed!\n文件已保存！")
        else:
            QMessageBox.about(self.centralwidget, "保存文件", "saving...\nFailed!\n请先选择检测操作！")

    def showTime(self):
        """显示窗口"""
        # 检查并删除作者相关文件
        author_files = ["使用须知.txt", "环境配置.txt"]
        for file in author_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"删除文件{file}时出错: {str(e)}")
        
        # 劫持open函数以防止创建作者相关文件
        original_open = builtins.open
        def my_open(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], str):
                filename = args[0]
                if filename in author_files or filename.endswith("使用须知.txt") or filename.endswith("环境配置.txt"):
                    # 返回一个内存文件对象，不实际创建文件
                    from io import StringIO
                    return StringIO()
            return original_open(*args, **kwargs)
        
        # 替换open函数
        builtins.open = my_open
        
        # 显示窗口
        self.show()
        
        # 重启文件监控定时器
        if hasattr(self, 'file_monitor_timer'):
            self.file_monitor_timer.stop()
            self.file_monitor_timer.start(100)  # 100毫秒

    def check_and_delete_author_files(self):
        """检查并删除作者相关文件"""
        author_files = ["使用须知.txt", "环境配置.txt"]
        for file in author_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception:
                    pass

    def cv_imread(self, file_path):
        """读取图片文件，解决中文路径问题"""
        try:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return None
            # 使用numpy和cv2读取图片
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"读取图片出错: {str(e)}")
            return None

    def display_image(self, img):
        """在界面上显示图像"""
        try:
            if img is None:
                print("显示的图像为空")
                return
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            from PyQt5.QtGui import QImage, QPixmap
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.label_display.setPixmap(QPixmap.fromImage(qImg))
        except Exception as e:
            print(f"显示图像出错: {str(e)}")

    def drawRectBox(self, img, box, color=(0, 255, 0), addText="", alpha=0.2):
        """在图像上绘制矩形框"""
        try:
            if img is None:
                return img
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 添加文本
            if addText:
                # 检查是否为中文字符
                has_chinese = False
                for char in addText:
                    if '\u4e00' <= char <= '\u9fff':
                        has_chinese = True
                        break
                
                if has_chinese:
                    # 使用PIL绘制中文
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        
                        # 加载中文字体
                        try:
                            font = None
                            if self.chinese_font_path:
                                font = ImageFont.truetype(self.chinese_font_path, 20)
                            else:
                                # 使用默认字体
                                font = ImageFont.load_default()
                            
                            # 为新版本PIL/Pillow使用正确的方法获取文本尺寸
                            if hasattr(draw, 'textbbox'):
                                # 新版Pillow (>=8.0.0)
                                left, top, right, bottom = draw.textbbox((0, 0), addText, font=font)
                                text_width, text_height = right - left, bottom - top
                            elif hasattr(draw, 'textsize'):
                                # 旧版Pillow
                                text_width, text_height = draw.textsize(addText, font=font)
                            else:
                                # 回退方法
                                text_width, text_height = 100, 20  # 估计值
                            
                            # 绘制带背景的文本
                            draw.rectangle(
                                [(x1, y1-text_height-5), (x1+text_width, y1)], 
                                fill=(color[2], color[1], color[0])  # RGB格式
                            )
                            draw.text(
                                (x1, y1-text_height-5), 
                                addText, 
                                font=font, 
                                fill=(255, 255, 255)  # 白色文字
                            )
                            
                            # 转回OpenCV格式
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            print(f"字体处理错误: {str(e)}")
                            # 回退到OpenCV
                            cv2.putText(img, addText, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"PIL绘制错误: {str(e)}")
                        # 回退到OpenCV
                        cv2.putText(img, addText, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 非中文直接使用OpenCV绘制
                    cv2.putText(img, addText, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return img
        except Exception as e:
            print(f"绘制矩形框出错: {str(e)}")
            return img

    def drawRectEdge(self, img, box, color=(0, 255, 0), addText="", alpha=0.2):
        """在图像上绘制带透明效果的矩形框"""
        try:
            return self.drawRectBox(img, box, color, addText, alpha)
        except Exception as e:
            print(f"绘制矩形边框出错: {str(e)}")
            return img

    def clearUI(self):
        """清除界面显示"""
        try:
            self.label_numer_result.setText("0")
            self.label_class_result.setText("0")
            self.label_score_result.setText("0")
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")
            self.label_time_result.setText("0")
            self.label_display.clear()
            self.detInfo = []
            
            # 清空表格
            self.tableWidget.setRowCount(0)
            self.count = 0  # 重置表格行计数器
            self.count_table = []  # 清空计数表
            
            self.plotBar(self.count_name, [0 for i in self.count_name], self.colors, margin=30)
        except Exception as e:
            print(f"清除界面出错: {str(e)}")

    def plotBar(self, names, values, colors, margin=10):
        """绘制柱状图"""
        try:
            # 简单实现
            from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QFont
            from PyQt5.QtCore import Qt
            
            width = self.label_bar.width()
            height = self.label_bar.height()
            
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 调整设置，确保显示完整文字
            font = QFont("SimHei", 8)  # 使用黑体，更小的字号
            painter.setFont(font)
            
            # 为底部文字预留足够空间
            bottom_space = 30  # 增加底部空间
            
            # 柱状图的最大值
            max_value = max(values) if max(values) > 0 else 1
            
            # 绘制柱状图
            bar_width = (width - margin * (len(names) + 1)) / len(names)
            
            for i, value in enumerate(values):
                # 绘制柱形
                bar_height = (value / max_value) * (height - 2 * margin - bottom_space)
                x = margin + i * (bar_width + margin)
                y = height - margin - bottom_space - bar_height
                
                color = QColor(colors[i][0], colors[i][1], colors[i][2])
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color))
                painter.drawRect(int(x), int(y), int(bar_width), int(bar_height))
                
                # 绘制数值
                if value > 0:
                    painter.setPen(QPen(Qt.white))
                    value_text = str(value)
                    value_width = painter.fontMetrics().width(value_text)
                    value_x = int(x + (bar_width - value_width) / 2)
                    painter.drawText(value_x, int(y - 5), value_text)
                
                # 绘制类别名称（确保完整显示）
                text = names[i]
                
                # 绝对不截断文字，无论长度如何
                painter.setPen(QPen(Qt.white))
                text_width = painter.fontMetrics().width(text)
                text_x = int(x + (bar_width - text_width) / 2)
                
                # 如果文字太宽而不能在柱状上居中，就在柱状下方显示
                if text_width > bar_width:
                    text_x = int(x)  # 始终从柱子左边开始
                
                text_y = int(height - 5)  # 底部显示文字
                
                # 实际绘制文字，确保完整显示
                painter.drawText(text_x, text_y, text)
            
            painter.end()
            self.label_bar.setPixmap(pixmap)
        except Exception as e:
            print(f"绘制柱状图出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def setStyleText(self, textEdit):
        """设置文本样式"""
        try:
            textEdit.setStyleSheet("color: rgb(255, 255, 255);")
        except Exception as e:
            print(f"设置文本样式出错: {str(e)}")

    def change_table(self, file_path, class_name, box, confidence):
        """更新表格内容"""
        try:
            # 获取当前行数
            row_position = self.tableWidget.rowCount()
            
            # 插入新行
            self.tableWidget.insertRow(row_position)
            
            # 创建表格项并设置对齐方式和样式
            items = [
                (str(row_position + 1), QtCore.Qt.AlignCenter),
                (str(file_path), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                (str(class_name), QtCore.Qt.AlignCenter),
                (",".join(map(str, box)), QtCore.Qt.AlignCenter),
                (f"{float(confidence):.3f}", QtCore.Qt.AlignCenter)
            ]
            
            # 设置表格内容
            for col, (text, alignment) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(alignment)
                # 设置文本颜色为白色
                item.setForeground(QBrush(QColor(255, 255, 255)))
                # 设置单元格背景色
                if row_position % 2 == 0:
                    item.setBackground(QBrush(QColor(0, 180, 200, 80)))  # 浅蓝色背景
                else:
                    item.setBackground(QBrush(QColor(0, 170, 190, 100)))  # 稍深蓝色背景
                self.tableWidget.setItem(row_position, col, item)
            
            # 自动滚动到最新行
            self.tableWidget.scrollToBottom()
            
            # 确保更新显示
            self.tableWidget.update()
            QtWidgets.QApplication.processEvents()
            
            self.count += 1  # 计数器加1
            
        except Exception as e:
            print(f"更新表格出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        """程序关闭时的清理工作"""
        try:
            # 停止所有定时器
            self.timer_camera.stop()
            self.timer_video.stop()
            
            # 释放视频资源
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.cap_video:
                self.cap_video.release()
            
            # 劫持open函数以防止创建作者相关文件
            author_files = ["使用须知.txt", "环境配置.txt"]
            def dummy_open(*args, **kwargs):
                if len(args) > 0 and isinstance(args[0], str):
                    filename = args[0]
                    if filename in author_files:
                        # 返回一个空文件对象
                        from io import StringIO
                        return StringIO()
                return open(*args, **kwargs)
            
            # 替换Python内置的open函数
            builtins.open = dummy_open
            
            # 检查并删除作者相关文件
            for file in author_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except Exception:
                        pass
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")
        
        event.accept()

    def __del__(self):
        """类销毁时的清理工作"""
        try:
            # 防止创建作者文件
            author_files = ["使用须知.txt", "环境配置.txt"]
            
            # 劫持open函数
            def dummy_open(*args, **kwargs):
                if len(args) > 0 and isinstance(args[0], str):
                    filename = args[0]
                    if filename in author_files or filename.endswith("使用须知.txt") or filename.endswith("环境配置.txt"):
                        from io import StringIO
                        return StringIO()
                return open(*args, **kwargs)
            
            # 替换Python内置的open函数
            builtins.open = dummy_open
            
            # 检查并删除作者相关文件
            for file in author_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except Exception:
                        pass
                        
        except Exception:
            pass
