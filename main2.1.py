import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import cv2
import numpy as np
import threading
import time
import os
import json
import csv
import subprocess
from PIL import Image
import tempfile
import re
# 新增导入用于图表显示
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 尝试导入PaddleOCR，如无法导入则设置标志
try:
    from paddleocr import PaddleOCR

    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False


OCR_MODE = "gemini"
DEFAULT_TEST_FOLDER = r"C:\Users\李墨\Desktop\test\6"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRESSURE_MIN_MPA = 0.0
PRESSURE_MAX_MPA = 10.0
GEMINI_DIGIT_CROP_REGIONS = [
    ("digit_wide", (0.20, 0.20, 0.95, 0.72)),
    ("digit_middle", (0.25, 0.22, 0.95, 0.65)),
    ("digit_tight", (0.30, 0.25, 0.92, 0.60)),
]
GEMINI_RETRY_PREPROCESS_METHODS = [
    "original",
    "enhanced",
    "lcd_sharp",
    "lcd_binary",
    "lcd_dark",
]


def read_image_cv2(image_path):
    """Read image paths with non-ASCII characters on Windows."""
    try:
        return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(image_path)


def format_pressure_value(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def normalize_pressure_number(raw_number):
    """Normalize OCR text such as 3695, 4MPa, or 4.50 MPa into an MPa float."""
    if raw_number is None:
        return None

    text = str(raw_number).strip()
    if not text:
        return None

    text = re.sub(r"(?i)\s*m\s*p\s*a", "", text)
    text = re.sub(r"[,;:_oO]", ".", text)
    text = text.replace(" ", "")
    text = re.sub(r"[^0-9.]", "", text)
    if not text:
        return None

    candidates = []
    if "." in text:
        parts = text.split(".")
        int_part = parts[0] or "0"
        dec_part = "".join(parts[1:])
        if dec_part:
            candidates.append(f"{int_part}.{dec_part}")
            if len(int_part) > 1:
                candidates.append(f"{int_part[-1]}.{dec_part}")
    else:
        digits = re.sub(r"\D", "", text)
        if not digits:
            return None
        if len(digits) >= 4:
            candidates.append(f"{digits[0]}.{digits[1:]}")
            candidates.append(f"{digits[-4]}.{digits[-3:]}")
        elif len(digits) == 3:
            candidates.append(f"{digits[0]}.{digits[1:]}")
        elif len(digits) == 2:
            candidates.append(f"{digits[0]}.{digits[1]}")
        else:
            candidates.append(digits)

    for candidate in candidates:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if PRESSURE_MIN_MPA <= value <= PRESSURE_MAX_MPA:
            return value
    return None


def extract_mpa_candidates(text, base_score=0.0):
    """Extract and score pressure candidates from OCR text."""
    if not text:
        return []

    normalized_text = re.sub(r"[,;:_oO]", ".", str(text))
    candidates = []
    for match in re.finditer(r"\d+\.\d+|\.\d+|\d+", normalized_text):
        raw = match.group(0)
        value = normalize_pressure_number(raw)
        if value is None:
            continue

        window_start = max(0, match.start() - 8)
        window_end = min(len(normalized_text), match.end() + 8)
        context = normalized_text[window_start:window_end]
        has_mpa = re.search(r"(?i)m\s*p\s*a", context) is not None

        score = float(base_score or 0.0)
        score += 40 if has_mpa else 0
        score += 20 if "." in raw else 0
        score += 12 if raw.isdigit() and len(raw) in (3, 4) else 0
        score += 10 if 3.0 <= value <= 5.0 else 0
        score += max(0, 6 - abs(value - 4.0))
        candidates.append({
            "raw": raw,
            "value": value,
            "text": format_pressure_value(value),
            "has_mpa": has_mpa,
            "score": score,
            "context": context,
        })
    return candidates


def choose_best_pressure_value(candidates):
    valid = [
        candidate for candidate in candidates
        if PRESSURE_MIN_MPA <= candidate.get("value", -1) <= PRESSURE_MAX_MPA
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: item.get("score", 0.0))


def extract_mpa_value(text, base_score=0.0):
    candidates = extract_mpa_candidates(text, base_score=base_score)
    return choose_best_pressure_value(candidates), candidates


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


class GeminiOCRProcessor:
    """Reusable OCR adapter based on the recognition strategy from 2_Gemini_2.py."""

    def __init__(self, lang='en', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.initialized = False
        self.ocr = None
        self.init_error = ""

        if not PADDLE_OCR_AVAILABLE:
            self.init_error = "PaddleOCR模块不可用"
            return

        self.ocr = self._create_paddle_ocr(lang)
        self.initialized = self.ocr is not None

    def _create_paddle_ocr(self, lang):
        use_gpu = os.getenv("PADDLEOCR_USE_GPU", "0").strip().lower() in ("1", "true", "yes", "on")
        tuned_kwargs = {
            "use_angle_cls": False,
            "lang": lang,
            "det_db_thresh": 0.05,
            "det_db_box_thresh": 0.2,
            "det_db_unclip_ratio": 1.9,
            "rec_image_shape": "3, 48, 636",
        }
        if use_gpu:
            tuned_kwargs.update({"use_gpu": True, "gpu_mem": 1000})

        configs = [
            tuned_kwargs,
            {key: value for key, value in tuned_kwargs.items() if key not in ("use_gpu", "gpu_mem")},
            {"use_textline_orientation": False, "lang": lang},
            {"lang": lang},
        ]

        last_error = ""
        for kwargs in configs:
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:
                last_error = str(exc)
        self.init_error = f"Gemini OCR初始化失败: {last_error}"
        return None

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def recognize_image(self, image_path):
        if not self.initialized:
            return self._error_result(self.init_error or "Gemini OCR未初始化")

        original_img = read_image_cv2(image_path)
        if original_img is None:
            return self._error_result("图片读取失败")

        methods_to_try = ["original", "thicken_lines", "binary", "invert", "enhanced"]
        best = self._recognize_with_methods(original_img, methods_to_try)
        if best:
            return self._success_result(best, "direct")

        retry_candidates = []
        for region_name, region in GEMINI_DIGIT_CROP_REGIONS:
            cropped_img = self._crop_image_region(original_img, region)
            retry_best = self._recognize_with_methods(
                cropped_img,
                GEMINI_RETRY_PREPROCESS_METHODS,
                method_prefix=region_name,
            )
            if retry_best:
                retry_candidates.append(retry_best)

        if retry_candidates:
            retry_candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            return self._success_result(retry_candidates[0], "retry")

        return self._error_result("Gemini OCR未识别到0-10 MPa读数")

    def extract_digits_from_image(self, image_path):
        return self.recognize_image(image_path)

    def _recognize_with_methods(self, img_array, methods, method_prefix=""):
        candidates = []
        for method in methods:
            try:
                if method == "original":
                    processed_img = img_array
                else:
                    processed_img = self._preprocess_image_memory(img_array, method)
                if processed_img is None:
                    continue

                raw_text, confidence = self._ocr_execute_memory(processed_img)
                if not raw_text:
                    continue

                best_value, value_candidates = extract_mpa_value(raw_text, base_score=confidence * 100)
                if not best_value:
                    continue

                method_name = f"{method_prefix}+{method}" if method_prefix else method
                candidates.append({
                    "method": method_name,
                    "raw_text": raw_text,
                    "confidence": confidence,
                    "pressure": best_value,
                    "candidates": value_candidates,
                    "score": best_value.get("score", 0.0),
                })
            except Exception:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return candidates[0]

    def _ocr_execute_memory(self, img_array):
        try:
            result = None
            ocr_error = None
            if hasattr(self.ocr, "ocr"):
                try:
                    result = self.ocr.ocr(img_array, cls=False)
                except TypeError:
                    try:
                        result = self.ocr.ocr(img_array)
                    except Exception as exc:
                        ocr_error = exc
                except Exception as exc:
                    ocr_error = exc
            if result is None and hasattr(self.ocr, "predict"):
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        temp_path = tmp.name
                    cv2.imwrite(temp_path, img_array)
                    result = self.ocr.predict(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            if result is None and ocr_error:
                raise ocr_error

            texts, scores = self._collect_texts_scores(result)
            if not texts:
                return "", 0.0

            combined_text = " ".join(texts)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return combined_text, avg_score
        except Exception:
            return "", 0.0

    def _collect_texts_scores(self, result):
        texts = []
        scores = []

        def add_text(text, score=0.0):
            if text is None:
                return
            text = str(text).strip()
            if not text:
                return
            texts.append(text)
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                scores.append(0.0)

        def walk(node):
            if isinstance(node, dict):
                rec_texts = node.get("rec_texts") or node.get("texts") or []
                rec_scores = node.get("rec_scores") or node.get("scores") or []
                for index, text in enumerate(rec_texts):
                    score = rec_scores[index] if index < len(rec_scores) else 0.0
                    add_text(text, score)
                return

            if isinstance(node, (list, tuple)):
                if len(node) >= 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
                    add_text(node[0], node[1])
                    return
                if (
                    len(node) >= 2
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) >= 2
                    and isinstance(node[1][0], str)
                ):
                    add_text(node[1][0], node[1][1])
                    return
                for child in node:
                    walk(child)

        walk(result)
        return texts, scores

    def _preprocess_image_memory(self, img_array, method="enhanced"):
        if img_array is None:
            return None

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array

        height, width = gray.shape
        target_size = 320
        if height < target_size or width < target_size:
            scale = max(target_size / height, target_size / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        processed = gray
        if method == "thicken_lines":
            gray_inv = cv2.bitwise_not(gray) if np.mean(gray) > 127 else gray
            kernel_v = np.ones((4, 1), np.uint8)
            processed = cv2.dilate(gray_inv, kernel_v, iterations=1)
            processed = cv2.bitwise_not(processed)
        elif method == "binary":
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 25, 2
            )
            processed = cv2.bitwise_not(processed)
        elif method == "invert":
            processed = cv2.bitwise_not(gray)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        elif method == "enhanced":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
        elif method == "lcd_sharp":
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            processed = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        elif method == "lcd_binary":
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            processed = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 5
            )
        elif method == "lcd_dark":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            boosted = clahe.apply(gray)
            _, mask = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel_v = np.ones((2, 1), np.uint8)
            mask = cv2.dilate(mask, kernel_v, iterations=1)
            processed = cv2.bitwise_not(mask)

        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    def _crop_image_region(self, img_array, region):
        height, width = img_array.shape[:2]
        left, top, right, bottom = region
        x1 = max(0, min(width - 1, int(width * left)))
        y1 = max(0, min(height - 1, int(height * top)))
        x2 = max(x1 + 1, min(width, int(width * right)))
        y2 = max(y1 + 1, min(height, int(height * bottom)))
        return img_array[y1:y2, x1:x2]

    def _success_result(self, best, status):
        pressure = best["pressure"]
        pressure_text = pressure["text"]
        return {
            "text": pressure_text,
            "raw_text": best.get("raw_text", ""),
            "details": best.get("candidates", []),
            "high_confidence_texts": [best.get("raw_text", "")],
            "high_confidence_scores": [best.get("confidence", 0.0)],
            "numeric_values": [pressure_text],
            "pressure_value": pressure.get("value"),
            "confidence": best.get("confidence", 0.0),
            "engine": "gemini",
            "status": status,
            "best_method": best.get("method", ""),
            "error": "",
        }

    def _error_result(self, message):
        return {
            "text": message,
            "raw_text": "",
            "details": [],
            "high_confidence_texts": [],
            "high_confidence_scores": [],
            "numeric_values": [],
            "pressure_value": None,
            "confidence": 0.0,
            "engine": "gemini",
            "status": "failed",
            "best_method": "",
            "error": message,
        }


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
        self.is_folder_recognizing = False
        self.stop_folder_requested = False
        self.folder_thread = None

        # OCR设置
        self.enable_ocr = tk.BooleanVar(value=True)
        self.ocr_mode = OCR_MODE
        self.gemini_ocr_processor = None
        self.paddle_ocr_processor = None
        self.ocr_lock = threading.Lock()

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
        """初始化优先OCR处理器和PaddleOCR fallback"""
        if not PADDLE_OCR_AVAILABLE:
            messagebox.showwarning("PaddleOCR不可用",
                                   "PaddleOCR模块未安装，请运行 'pip install paddleocr' 安装\nOCR功能将不可用")
            self.enable_ocr.set(False)
            return

        init_errors = []

        try:
            self.gemini_ocr_processor = GeminiOCRProcessor(
                lang='en',
                confidence_threshold=0.5
            )
            if not self.gemini_ocr_processor.initialized:
                init_errors.append(self.gemini_ocr_processor.init_error or "Gemini OCR初始化失败")
        except Exception as e:
            self.gemini_ocr_processor = None
            init_errors.append(f"Gemini OCR初始化失败: {str(e)}")

        try:
            self.paddle_ocr_processor = PaddleOCRProcessor(
                use_textline_orientation=True,
                lang='ch',  # 使用中文模型，能更好识别数字
                confidence_threshold=0.5
            )

            # 检查处理器是否成功初始化
            if not hasattr(self.paddle_ocr_processor, 'initialized') or not self.paddle_ocr_processor.initialized:
                init_errors.append("PaddleOCR fallback初始化失败")
        except Exception as e:
            self.paddle_ocr_processor = None
            init_errors.append(f"PaddleOCR fallback初始化失败: {str(e)}")

        gemini_ready = bool(self.gemini_ocr_processor and self.gemini_ocr_processor.initialized)
        paddle_ready = bool(self.paddle_ocr_processor and getattr(self.paddle_ocr_processor, 'initialized', False))
        if not gemini_ready and not paddle_ready:
            self.enable_ocr.set(False)
            messagebox.showwarning("OCR初始化失败",
                                   "Gemini OCR和PaddleOCR fallback均不可用，OCR功能将不可用")
        elif not gemini_ready:
            self.status_var.set("Gemini OCR不可用，已使用PaddleOCR fallback")
        elif init_errors:
            self.status_var.set("Gemini OCR已启用，PaddleOCR fallback不可用")

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

        self.folder_btn = tk.Button(button_frame, text="选择文件夹识别",
                                    command=self.select_folder_and_recognize, height=2)
        self.folder_btn.pack(fill=tk.X, pady=(8, 5))

        self.stop_folder_btn = tk.Button(button_frame, text="停止文件夹识别",
                                         command=self.stop_folder_recognition,
                                         state=tk.DISABLED, height=2)
        self.stop_folder_btn.pack(fill=tk.X)

        result_frame = tk.LabelFrame(right_frame, text="OCR识别结果")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        result_scroll = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化OCR设置状态
        self.toggle_ocr_settings()

    def update_confidence_threshold(self, value):
        """更新置信度阈值"""
        if self.gemini_ocr_processor:
            self.gemini_ocr_processor.set_confidence_threshold(float(value))
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

    def _extract_digits_from_image_legacy(self, image_path):
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

    def extract_digits_from_image(self, image_path):
        lock = getattr(self, "ocr_lock", None)
        if lock:
            with lock:
                return self._extract_digits_from_image_unlocked(image_path)
        return self._extract_digits_from_image_unlocked(image_path)

    def _extract_digits_from_image_unlocked(self, image_path):
        """Prefer the 2_Gemini_2.py OCR strategy and fallback to the old PaddleOCR path."""
        if not self.enable_ocr.get():
            return {"text": "OCR功能未启用", "numeric_values": [], "pressure_value": None}

        errors = []
        if self.gemini_ocr_processor and self.gemini_ocr_processor.initialized:
            try:
                result = self.gemini_ocr_processor.recognize_image(image_path)
                if isinstance(result, dict) and result.get("pressure_value") is not None:
                    return result
                if isinstance(result, dict) and result.get("error"):
                    errors.append(result.get("error"))
            except Exception as e:
                errors.append(f"Gemini OCR失败: {str(e)}")

        if self.paddle_ocr_processor and getattr(self.paddle_ocr_processor, 'initialized', False):
            try:
                result = self.paddle_ocr_processor.extract_digits_from_image(image_path)
                if isinstance(result, dict):
                    normalized = self._normalize_paddle_result(result)
                    if normalized.get("pressure_value") is not None:
                        if errors:
                            normalized["text"] = f"{normalized['text']} (fallback)"
                        return normalized
                    errors.append(normalized.get("error") or "PaddleOCR fallback未识别到有效MPa读数")
                else:
                    errors.append("PaddleOCR fallback结果格式异常")
            except Exception as e:
                errors.append(f"PaddleOCR fallback失败: {str(e)}")

        error_text = "OCR识别失败"
        if errors:
            error_text = f"{error_text}: {'; '.join(errors[:2])}"
        return {
            "text": error_text,
            "numeric_values": [],
            "pressure_value": None,
            "engine": "none",
            "error": error_text,
        }

    def _normalize_paddle_result(self, result):
        raw_text = result.get("text", "")
        source_text = " ".join(
            [raw_text] + [str(item) for item in result.get("numeric_values", [])]
        )
        best_value, candidates = extract_mpa_value(source_text)
        if not best_value:
            return {
                "text": raw_text or "PaddleOCR fallback未识别到有效MPa读数",
                "raw_text": raw_text,
                "details": result.get("details", []),
                "numeric_values": [],
                "pressure_value": None,
                "confidence": 0.0,
                "engine": "paddle_fallback",
                "error": "PaddleOCR fallback未识别到有效MPa读数",
            }

        pressure_text = best_value["text"]
        return {
            "text": pressure_text,
            "raw_text": raw_text,
            "details": candidates or result.get("details", []),
            "high_confidence_texts": result.get("high_confidence_texts", []),
            "high_confidence_scores": result.get("high_confidence_scores", []),
            "numeric_values": [pressure_text],
            "pressure_value": best_value["value"],
            "confidence": max(result.get("high_confidence_scores", [0.0]) or [0.0]),
            "engine": "paddle_fallback",
            "error": "",
        }

    def _update_chart_legacy(self, timestamp_str, values):
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

    def update_chart(self, timestamp_str, values):
        """Schedule chart data updates on the Tkinter main thread."""
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = datetime.now()

        parsed_values = []
        for value_item in values:
            try:
                parsed_values.append(float(value_item))
            except (TypeError, ValueError):
                continue

        if parsed_values:
            self.root.after(0, self._append_chart_values, timestamp, parsed_values)

    def _append_chart_values(self, timestamp, values):
        for value in values:
            self.chart_data['timestamps'].append(timestamp)
            self.chart_data['values'].append(value)
        self._update_chart_ui()

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

    def select_folder_and_recognize(self):
        if self.is_folder_recognizing:
            messagebox.showinfo("提示", "文件夹识别正在进行中")
            return
        if not self.enable_ocr.get():
            messagebox.showwarning("OCR不可用", "OCR功能未启用，无法进行文件夹识别")
            return

        initial_dir = DEFAULT_TEST_FOLDER if os.path.isdir(DEFAULT_TEST_FOLDER) else os.getcwd()
        folder_path = filedialog.askdirectory(
            initialdir=initial_dir,
            title="选择包含图片的文件夹"
        )
        if not folder_path:
            return

        self.is_folder_recognizing = True
        self.stop_folder_requested = False
        self.folder_btn.config(state=tk.DISABLED)
        self.stop_folder_btn.config(state=tk.NORMAL)
        self.append_ocr_result_to_gui({
            "image_name": "folder",
            "image_path": folder_path,
            "raw_text": "开始文件夹识别",
            "mpa_value": "",
            "engine": "",
            "success": True,
            "error": "",
        })
        self.set_status(f"开始识别文件夹: {folder_path}")

        self.folder_thread = threading.Thread(
            target=self.recognize_folder_images,
            args=(folder_path,),
            daemon=True
        )
        self.folder_thread.start()

    def stop_folder_recognition(self):
        if self.is_folder_recognizing:
            self.stop_folder_requested = True
            self.set_status("正在停止文件夹识别...")

    def recognize_folder_images(self, folder_path):
        results = []
        stopped = False
        try:
            image_files = self._list_folder_images(folder_path)
            total = len(image_files)
            if total == 0:
                self.root.after(0, self._finish_folder_recognition, 0, "", False, "文件夹中没有支持的图片文件")
                return

            for index, image_path in enumerate(image_files, start=1):
                if self.stop_folder_requested:
                    stopped = True
                    break

                image_name = os.path.basename(image_path)
                self.set_status(f"正在识别 {index}/{total}: {image_name}")
                result = self.recognize_single_image(image_path)
                result["index"] = index
                result["total"] = total
                results.append(result)
                self.append_ocr_result_to_gui(result)
                if result.get("success") and result.get("mpa_value") is not None:
                    self.update_trend_from_value(result["mpa_value"])

            result_file = self.save_folder_ocr_results(results, folder_path) if results else ""
            message = "文件夹识别已停止" if stopped else "文件夹识别完成"
            self.root.after(0, self._finish_folder_recognition, len(results), result_file, stopped, message)
        except Exception as e:
            self.root.after(0, self._finish_folder_recognition, len(results), "", stopped, f"文件夹识别失败: {str(e)}")

    def _list_folder_images(self, folder_path):
        if not os.path.isdir(folder_path):
            return []

        image_files = []
        for root_dir, _, filenames in os.walk(folder_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root_dir, filename))
        return sorted(image_files, key=lambda path: os.path.basename(path).lower())

    def recognize_single_image(self, image_path):
        image_name = os.path.basename(image_path)
        try:
            ocr_result = self.extract_digits_from_image(image_path)
            raw_text = ocr_result.get("raw_text") or ocr_result.get("text", "")
            mpa_value = ocr_result.get("pressure_value")
            numeric_values = ocr_result.get("numeric_values", [])
            if mpa_value is None and numeric_values:
                try:
                    mpa_value = float(numeric_values[0])
                except (TypeError, ValueError):
                    mpa_value = None

            engine = self._format_engine_name(ocr_result.get("engine", ""))
            error = ocr_result.get("error", "")
            success = mpa_value is not None
            if not success and not error:
                error = "未识别到有效MPa读数"

            return {
                "image_name": image_name,
                "image_path": image_path,
                "raw_text": raw_text,
                "mpa_value": mpa_value,
                "engine": engine,
                "success": success,
                "error": error,
                "best_method": ocr_result.get("best_method", ""),
                "display_text": ocr_result.get("text", ""),
            }
        except Exception as e:
            return {
                "image_name": image_name,
                "image_path": image_path,
                "raw_text": "",
                "mpa_value": None,
                "engine": "",
                "success": False,
                "error": str(e),
                "best_method": "",
                "display_text": "",
            }

    def save_folder_ocr_results(self, results, folder_path):
        output_path = os.path.join(folder_path, "folder_ocr_results.csv")
        fieldnames = [
            "image_name",
            "image_path",
            "raw_text",
            "mpa_value",
            "engine",
            "success",
            "error",
            "best_method",
        ]
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({key: result.get(key, "") for key in fieldnames})
        return output_path

    def append_ocr_result_to_gui(self, result):
        if threading.current_thread() is threading.main_thread():
            self._append_ocr_result_to_gui_now(result)
        else:
            self.root.after(0, self._append_ocr_result_to_gui_now, result)

    def _append_ocr_result_to_gui_now(self, result):
        if not hasattr(self, "result_text"):
            return
        status = "成功" if result.get("success") else "失败"
        mpa_value = result.get("mpa_value")
        try:
            mpa_text = "" if mpa_value in (None, "") else format_pressure_value(float(mpa_value))
        except (TypeError, ValueError):
            mpa_text = str(mpa_value)
        line = (
            f"[{status}] {result.get('image_name', '')} "
            f"MPa={mpa_text} 引擎={result.get('engine', '')} "
            f"文本={result.get('raw_text', '')}"
        )
        if result.get("error"):
            line += f" 错误={result.get('error')}"
        self.result_text.insert(tk.END, line + "\n")
        self.result_text.see(tk.END)

    def update_trend_from_value(self, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        self.update_chart(time.strftime("%Y%m%d_%H%M%S"), [numeric_value])

    def _finish_folder_recognition(self, count, result_file, stopped, message):
        self.is_folder_recognizing = False
        self.stop_folder_requested = False
        self.folder_btn.config(state=tk.NORMAL)
        self.stop_folder_btn.config(state=tk.DISABLED)
        status_text = f"{message}，已处理 {count} 张图片"
        if result_file:
            status_text += f"，结果: {result_file}"
        self.set_status(status_text)
        self.append_ocr_result_to_gui({
            "image_name": "folder",
            "image_path": result_file,
            "raw_text": status_text,
            "mpa_value": "",
            "engine": "",
            "success": not stopped,
            "error": "",
        })

    def _format_engine_name(self, engine):
        if engine == "gemini":
            return "Gemini"
        if engine == "paddle_fallback":
            return "PaddleOCR"
        return engine or ""

    def set_status(self, text):
        if threading.current_thread() is threading.main_thread():
            self.status_var.set(text)
        else:
            self.root.after(0, self.status_var.set, text)

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

                    self.append_ocr_result_to_gui({
                        "image_name": filename,
                        "image_path": filepath,
                        "raw_text": ocr_result.get("raw_text") or ocr_result_text,
                        "mpa_value": ocr_result.get("pressure_value"),
                        "engine": self._format_engine_name(ocr_result.get("engine", "")),
                        "success": bool(numeric_values),
                        "error": ocr_result.get("error", ""),
                        "best_method": ocr_result.get("best_method", ""),
                    })

                # 更新状态
                status_text = f"已拍摄 {photo_count} 张照片，最后保存: {filename}"
                if ocr_result_text:
                    status_text += f", 识别结果: {ocr_result_text}"
                self.set_status(status_text)

            # 等待指定间隔
            for _ in range(int(self.capture_interval * 10)):
                if not self.is_capturing:
                    break
                time.sleep(0.1)

        self.set_status("拍照已停止")

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

        self.set_status("开始拍照...")

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
        app.stop_folder_requested = True
        if hasattr(app, 'cap'):
            app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
