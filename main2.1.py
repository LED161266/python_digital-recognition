import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import cv2
import threading
import time
import os
import json
import subprocess
from PIL import Image
import tempfile
import re
# 新增导入用于图表显示
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime

# 尝试导入PaddleOCR，如无法导入则设置标志
try:
    from paddleocr import PaddleOCR

    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False


class PaddleOcrJson:
    """
    Python 调用 PaddleOCR-json 的示例类
    基于 https://github.com/hiroi-sora/PaddleOCR-json
    """

    def __init__(self, exe_path: str, args: dict = None):
        """
        初始化 OCR 引擎

        :param exe_path: PaddleOCR_json.exe 的完整路径
        :param args: 启动参数字典，默认为 None
        """
        self.exe_path = exe_path
        self.args = args or {}  # 默认参数字典
        self.process = None

        # 构建命令列表
        self.cmd = [self.exe_path]
        for key, value in self.args.items():
            self.cmd.extend([f"-{key}", str(value)])

        # 启动引擎进程
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # 使用文本模式
                encoding="utf-8",
                bufsize=1,  # 行缓冲
                cwd=os.path.dirname(self.exe_path)  # 在工作目录启动
            )
        except Exception as e:
            raise Exception(f"Failed to start PaddleOCR-json process: {e}")

    def ocr(self, image_input):
        """
        对图片进行 OCR 识别

        :param image_input: 可以是图片路径（字符串）或 PIL.Image 对象
        :return: 识别结果的字典
        """
        if isinstance(image_input, str):
            # 输入是文件路径
            command_obj = {"image_path": image_input}
        elif isinstance(image_input, Image.Image):
            # 输入是 PIL Image 对象，保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                image_input.save(temp_path, "PNG")
                command_obj = {"image_path": temp_path}
        else:
            raise ValueError("Unsupported image input type. Use path string or PIL.Image.")

        # 发送命令
        try:
            command_str = json.dumps(command_obj, ensure_ascii=False) + "\n"
            self.process.stdin.write(command_str)
            self.process.stdin.flush()

            # 读取返回
            result_str = self.process.stdout.readline().strip()
            result = json.loads(result_str)

            # 清理临时文件
            if isinstance(image_input, Image.Image):
                os.unlink(temp_path)

            return result

        except Exception as e:
            # 确保临时文件被清理
            if isinstance(image_input, Image.Image) and 'temp_path' in locals():
                os.unlink(temp_path)
            raise Exception(f"OCR processing failed: {e}")

    def close(self):
        """关闭引擎进程"""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=3)
            self.process = None

    def __enter__(self):
        """支持上下文管理器 with 语法"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭引擎"""
        self.close()


class PaddleOCRProcessor:
    def __init__(self, use_textline_orientation=True, lang='en', confidence_threshold=0.5):
        """
        初始化PaddleOCR处理器

        :param use_textline_orientation: 是否使用文本方向检测
        :param lang: 语言设置
        :param confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold

        if not PADDLE_OCR_AVAILABLE:
            print("PaddleOCR模块不可用，请先安装: pip install paddleocr")
            self.initialized = False
            return

        try:
            from paddleocr import PaddleOCR  # 再次导入以确保可用
            self.ocr = PaddleOCR(
                use_textline_orientation=use_textline_orientation,
                lang=lang,
                # show_log=False  # 关闭详细日志输出
            )
            self.initialized = True
        except Exception as e:
            print(f"PaddleOCR初始化失败: {e}")
            self.initialized = False

    def extract_digits_from_image(self, image_path):
        """
        使用PaddleOCR从图像中提取数字和文本，优化支持小数识别

        :param image_path: 图像文件路径
        :return: 包含识别结果和详细信息的字典
        """
        if not self.initialized:
            return {
                'text': "PaddleOCR未正确初始化",
                'details': [],
                'high_confidence_texts': [],
                'numeric_values': []  # 新增：保存识别到的数值（包括小数）
            }

        try:
            # 使用PaddleOCR进行识别
            result = self.ocr.predict(image_path)

            if not result:  # 确保结果不为空
                return {
                    'text': "未识别到内容",
                    'details': [],
                    'high_confidence_texts': [],
                    'numeric_values': []
                }

            # 获取第一个结果字典
            data = result[0]

            # 获取所有识别到的文本和对应的置信度
            all_texts = data['rec_texts']
            all_scores = data['rec_scores']

            # 如果没有识别到任何内容
            if not all_texts:
                return {
                    'text': "未识别到内容",
                    'details': [],
                    'high_confidence_texts': [],
                    'numeric_values': []
                }

            # 提取所有数字和小数（优化点1：支持小数识别）
            numeric_values = []
            for text in all_texts:
                # 使用正则表达式提取整数和小数
                # 这个正则表达式可以匹配整数、小数（包括.3这样的形式）
                found_numbers = re.findall(r'\d+\.\d+|\.\d+|\d+', text)
                for num_str in found_numbers:
                    # 转换为浮点数进行标准化
                    try:
                        num = float(num_str)
                        # 保存原始字符串形式，确保格式正确
                        numeric_values.append(num_str)
                    except ValueError:
                        continue

            # 优化点：检测可能丢失小数点的数字（如3695 -> 3.695）
            # 这里假设如果一个整数大于等于1000且小于10000，它可能是一个带有小数点的数字
            # 例如：3695 -> 3.695, 4567 -> 4.567等
            corrected_numeric_values = []
            for num_str in numeric_values:
                # 检查是否为纯整数且长度为4位
                if '.' not in num_str and num_str.isdigit() and len(num_str) == 4:
                    # 将整数转换为小数形式
                    try:
                        # 插入小数点，例如：3695 -> 3.695
                        corrected_num_str = f"{num_str[0]}.{num_str[1:]}"
                        # 验证转换是否有效
                        float(corrected_num_str)
                        corrected_numeric_values.append(corrected_num_str)
                    except ValueError:
                        # 如果转换失败，保留原始值
                        corrected_numeric_values.append(num_str)
                else:
                    # 对于已经包含小数点的数字或不符合条件的数字，保留原始值
                    corrected_numeric_values.append(num_str)

            # 根据置信度过滤结果
            high_confidence_texts = []
            high_confidence_scores = []

            for text, score in zip(all_texts, all_scores):
                if score > self.confidence_threshold:  # 使用设置的阈值
                    high_confidence_texts.append(text)
                    high_confidence_scores.append(score)

            # 构建详细信息列表
            details = []
            for i, (text, score) in enumerate(zip(all_texts, all_scores)):
                detail = {
                    'text': text,
                    'confidence': score,
                    'position': i + 1
                }
                details.append(detail)

            # 合并所有识别到的数字和小数（使用修正后的值）
            combined_text = ' '.join(corrected_numeric_values) if corrected_numeric_values else "未识别到数字"

            return {
                'text': combined_text,
                'details': details,
                'high_confidence_texts': high_confidence_texts,
                'high_confidence_scores': high_confidence_scores,
                'numeric_values': corrected_numeric_values  # 返回修正后的数值列表
            }

        except Exception as e:
            return {
                'text': f"OCR错误: {str(e)}",
                'details': [],
                'high_confidence_texts': [],
                'numeric_values': []
            }

    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = threshold


class PhotoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("自动拍照与数字识别系统")
        self.root.geometry("1000x700")  # 稍微增大窗口以容纳图表

        # 拍照控制变量
        self.is_capturing = False
        self.capture_interval = 5  # 默认5秒
        self.save_path = os.getcwd()  # 默认当前目录
        self.text_save_path = os.getcwd()  # 文本保存路径
        self.capture_thread = None

        # OCR设置
        self.enable_ocr = tk.BooleanVar(value=True)
        self.paddle_ocr_processor = None

        # 图表相关数据（新增）
        self.chart_data = {
            'timestamps': [],  # 时间戳列表
            'values': []       # 对应的MPa值列表
        }

        # 创建界面
        self.create_widgets()

        # 初始化PaddleOCR
        self.init_paddle_ocr()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法访问摄像头")
            return

        # 开始预览
        self.update_preview()

    def init_paddle_ocr(self):
        """初始化PaddleOCR处理器"""
        if not PADDLE_OCR_AVAILABLE:
            messagebox.showwarning("PaddleOCR不可用",
                                   "PaddleOCR模块未安装，请运行 'pip install paddleocr' 安装\nOCR功能将不可用")
            self.enable_ocr.set(False)
            return

        try:
            self.paddle_ocr_processor = PaddleOCRProcessor(
                use_textline_orientation=True,
                lang='ch',  # 使用中文模型，能更好识别数字
                confidence_threshold=0.5
            )

            # 检查处理器是否成功初始化
            if not hasattr(self.paddle_ocr_processor, 'initialized') or not self.paddle_ocr_processor.initialized:
                self.enable_ocr.set(False)
                messagebox.showwarning("PaddleOCR初始化失败",
                                       "PaddleOCR初始化失败，OCR功能将不可用")
        except Exception as e:
            messagebox.showwarning("PaddleOCR初始化警告",
                                   f"PaddleOCR初始化失败: {str(e)}\nOCR功能将不可用")
            self.enable_ocr.set(False)

    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧预览和图表区域
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 摄像头预览区域
        preview_frame = tk.LabelFrame(left_frame, text="摄像头预览")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = tk.Label(preview_frame, text="摄像头未连接", bg="black")
        self.preview_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # 图表区域（新增，放置在预览区域下方）
        chart_frame = tk.LabelFrame(left_frame, text="MPa数值趋势图")
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # 设置Matplotlib字体支持中文（解决Glyph警告）
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 创建图表
        self.figure = plt.Figure(figsize=(6, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('MPa数值随时间变化')
        self.ax.set_xlabel('拍摄时间')
        self.ax.set_ylabel('MPa')
        self.ax.grid(True)
        # 设置x轴时间格式
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.figure.autofmt_xdate(rotation=45)
        self.figure.tight_layout()

        # 创建Tkinter画布用于显示图表
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右侧控制区域
        right_frame = tk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # 拍照设置
        settings_frame = tk.LabelFrame(right_frame, text="拍照设置")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # 间隔设置
        interval_frame = tk.Frame(settings_frame)
        interval_frame.pack(fill=tk.X, pady=5)

        tk.Label(interval_frame, text="拍照间隔(秒):").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value=str(self.capture_interval))
        interval_spinbox = tk.Spinbox(interval_frame, from_=1, to=3600,
                                      textvariable=self.interval_var, width=10)
        interval_spinbox.pack(side=tk.RIGHT, padx=5)

        # 图片保存路径设置
        img_path_frame = tk.Frame(settings_frame)
        img_path_frame.pack(fill=tk.X, pady=5)

        tk.Label(img_path_frame, text="图片保存路径:").pack(anchor=tk.W)

        img_path_btn_frame = tk.Frame(img_path_frame)
        img_path_btn_frame.pack(fill=tk.X, pady=2)

        self.img_path_var = tk.StringVar(value=self.save_path)
        img_path_entry = tk.Entry(img_path_btn_frame, textvariable=self.img_path_var)
        img_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        img_browse_btn = tk.Button(img_path_btn_frame, text="浏览",
                                   command=lambda: self.browse_path(self.img_path_var, "图片"))
        img_browse_btn.pack(side=tk.RIGHT)

        # OCR设置区域
        ocr_frame = tk.LabelFrame(right_frame, text="PaddleOCR数字识别设置")
        ocr_frame.pack(fill=tk.X, pady=(0, 10))

        # OCR启用复选框
        ocr_check = tk.Checkbutton(ocr_frame, text="启用数字识别",
                                   variable=self.enable_ocr, command=self.toggle_ocr_settings)
        ocr_check.pack(anchor=tk.W, pady=5)

        # 置信度阈值设置
        confidence_frame = tk.Frame(ocr_frame)
        confidence_frame.pack(fill=tk.X, pady=5)

        tk.Label(confidence_frame, text="置信度阈值:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(confidence_frame, from_=0.1, to=1.0,
                                    resolution=0.1, orient=tk.HORIZONTAL,
                                    variable=self.confidence_var,
                                    command=self.update_confidence_threshold)
        confidence_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 文本保存路径设置
        self.text_path_frame = tk.Frame(ocr_frame)
        self.text_path_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.text_path_frame, text="文本保存路径:").pack(anchor=tk.W)

        text_path_btn_frame = tk.Frame(self.text_path_frame)
        text_path_btn_frame.pack(fill=tk.X, pady=2)

        self.text_path_var = tk.StringVar(value=self.text_save_path)
        text_path_entry = tk.Entry(text_path_btn_frame, textvariable=self.text_path_var)
        text_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        text_browse_btn = tk.Button(text_path_btn_frame, text="浏览",
                                    command=lambda: self.browse_path(self.text_path_var, "文本"))
        text_browse_btn.pack(side=tk.RIGHT)

        # 文本文件名设置
        text_filename_frame = tk.Frame(ocr_frame)
        text_filename_frame.pack(fill=tk.X, pady=5)

        tk.Label(text_filename_frame, text="文本文件名:").pack(side=tk.LEFT)
        self.text_filename_var = tk.StringVar(value="recognized_digits.txt")
        text_filename_entry = tk.Entry(text_filename_frame, textvariable=self.text_filename_var, width=15)
        text_filename_entry.pack(side=tk.RIGHT, padx=(5, 0))

        # 按钮区域
        button_frame = tk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.start_btn = tk.Button(button_frame, text="开始拍照",
                                   command=self.start_capture, bg="green", fg="white", height=2)
        self.start_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_btn = tk.Button(button_frame, text="停止拍照",
                                  command=self.stop_capture, bg="red", fg="white",
                                  state=tk.DISABLED, height=2)
        self.stop_btn.pack(fill=tk.X)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化OCR设置状态
        self.toggle_ocr_settings()

    def update_confidence_threshold(self, value):
        """更新置信度阈值"""
        if self.paddle_ocr_processor:
            self.paddle_ocr_processor.set_confidence_threshold(float(value))

    def toggle_ocr_settings(self):
        """根据OCR启用状态显示或隐藏OCR设置"""
        if self.enable_ocr.get():
            self.text_path_frame.pack(fill=tk.X, pady=5)
        else:
            self.text_path_frame.pack_forget()

    def browse_path(self, path_var, path_type):
        path = filedialog.askdirectory(initialdir=path_var.get())
        if path:
            path_var.set(path)
            if path_type == "图片":
                self.save_path = path
            else:
                self.text_save_path = path

    def update_preview(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 转换颜色空间 BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整预览大小
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                # 转换为PIL图像
                pil_img = Image.fromarray(frame_rgb)
                # 转换为Tkinter图像
                tk_img = ImageTk.PhotoImage(pil_img)

                # 更新预览
                self.preview_label.configure(image=tk_img)
                self.preview_label.image = tk_img

        # 每隔30毫秒更新一次预览
        self.root.after(30, self.update_preview)

    def extract_digits_from_image(self, image_path):
        """使用PaddleOCR从图像中提取数字（支持小数）"""
        if not self.enable_ocr.get() or not self.paddle_ocr_processor:
            return {"text": "OCR功能未启用", "numeric_values": []}
        try:
            result = self.paddle_ocr_processor.extract_digits_from_image(image_path)
            # 确保返回的是包含text和numeric_values的字典
            if isinstance(result, dict):
                return result
            else:
                return {"text": "OCR结果格式异常", "numeric_values": []}
        except Exception as e:
            return {"text": f"OCR处理错误: {str(e)}", "numeric_values": []}

    def update_chart(self, timestamp_str, values):
        """更新图表显示"""
        # 转换时间戳字符串为datetime对象
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            # 如果时间格式不正确，则使用当前时间
            timestamp = datetime.now()

        # 过滤并添加有效的数值
        valid_values_added = False
        for value_str in values:
            try:
                value = float(value_str)
                # 假设这些数值代表MPa
                self.chart_data['timestamps'].append(timestamp)
                self.chart_data['values'].append(value)
                valid_values_added = True
            except ValueError:
                continue

        # 如果有有效数据添加，则更新图表
        if valid_values_added:
            # 在主线程中更新UI
            self.root.after(0, self._update_chart_ui)

    def _update_chart_ui(self):
        """在主线程中更新图表UI"""
        # 清除现有图表
        self.ax.clear()

        # 重新绘制图表
        self.ax.plot(self.chart_data['timestamps'], self.chart_data['values'], 'o-', color='blue')
        self.ax.set_title('MPa数值随时间变化')
        self.ax.set_xlabel('拍摄时间')
        self.ax.set_ylabel('MPa')
        self.ax.grid(True)

        # 修改时间格式以支持超过24小时
        # 首先检查数据范围是否超过24小时
        if len(self.chart_data['timestamps']) >= 2:
            time_diff = self.chart_data['timestamps'][-1] - self.chart_data['timestamps'][0]
            # 如果数据跨度超过24小时，使用自定义格式
            if time_diff.total_seconds() > 24 * 3600:
                # 使用天数、小时、分钟格式
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d天%H:%M:%S'))
            else:
                # 仍然使用小时、分钟、秒格式
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        else:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        self.figure.autofmt_xdate(rotation=45)
        self.figure.tight_layout()

        # 更新画布
        self.chart_canvas.draw()

    def capture_photos(self):
        photo_count = 0
        while self.is_capturing:
            # 拍照
            ret, frame = self.cap.read()
            if ret:
                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"photo_{timestamp}_{photo_count:04d}.jpg"
                filepath = os.path.join(self.save_path, filename)

                # 保存照片
                cv2.imwrite(filepath, frame)
                photo_count += 1

                # 如果启用了OCR，识别数字
                ocr_result_text = ""
                if self.enable_ocr.get():
                    ocr_result = self.extract_digits_from_image(filepath)
                    ocr_result_text = ocr_result.get('text', '')
                    numeric_values = ocr_result.get('numeric_values', [])

                    # 保存识别结果到文本文件
                    text_filepath = os.path.join(self.text_save_path, self.text_filename_var.get())
                    with open(text_filepath, "a", encoding="utf-8") as f:
                        f.write(f"{timestamp} - {filename}: {ocr_result_text}\n")

                    # 如果识别到有效数值，更新图表（功能2）
                    if numeric_values:
                        # 在主线程中更新图表
                        self.update_chart(timestamp, numeric_values)

                # 更新状态
                status_text = f"已拍摄 {photo_count} 张照片，最后保存: {filename}"
                if ocr_result_text:
                    status_text += f", 识别结果: {ocr_result_text}"
                self.status_var.set(status_text)

            # 等待指定间隔
            for _ in range(int(self.capture_interval * 10)):
                if not self.is_capturing:
                    break
                time.sleep(0.1)

        self.status_var.set("拍照已停止")

    def start_capture(self):
        try:
            self.capture_interval = float(self.interval_var.get())
            if self.capture_interval <= 0:
                messagebox.showerror("错误", "拍照间隔必须大于0")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            return

        # 检查图片保存路径
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path)
            except:
                messagebox.showerror("错误", "无法创建图片保存目录")
                return

        # 如果启用了OCR，检查文本保存路径
        if self.enable_ocr.get():
            if not os.path.exists(self.text_save_path):
                try:
                    os.makedirs(self.text_save_path)
                except:
                    messagebox.showerror("错误", "无法创建文本保存目录")
                    return

            # 检查文本文件名
            if not self.text_filename_var.get().strip():
                messagebox.showerror("错误", "请输入有效的文本文件名")
                return

        self.is_capturing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # 在单独线程中运行拍照功能
        self.capture_thread = threading.Thread(target=self.capture_photos)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.status_var.set("开始拍照...")

    def stop_capture(self):
        self.is_capturing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    root = tk.Tk()
    app = PhotoCaptureApp(root)

    def on_closing():
        app.is_capturing = False
        if hasattr(app, 'cap'):
            app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()