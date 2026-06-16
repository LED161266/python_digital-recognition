import pandas as pd
import os
import cv2
import sys
import re
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
from datetime import datetime

# --- 环境配置 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置模型根目录
model_dir = "D:\\paddle_models"

# 1. 创建模型目录
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 2. 创建虚拟的桌面目录，防止文件对话框报错
desktop_dir = os.path.join(model_dir, "Desktop")
if not os.path.exists(desktop_dir):
    os.makedirs(desktop_dir)

# 3. 修改环境变量
os.environ["USERPROFILE"] = model_dir

CONFIG_FILE = "ocr_config.json"

OCR_STATUS_DIRECT = "原始识别"
OCR_STATUS_RETRY = "二次识别"
OCR_STATUS_FILLED = "补值"
OCR_STATUS_FAILED = "失败"

DIGIT_CROP_REGIONS = [
    ("数字区域-宽", (0.20, 0.20, 0.95, 0.72)),
    ("数字区域-中", (0.25, 0.22, 0.95, 0.65)),
    ("数字区域-紧", (0.30, 0.25, 0.92, 0.60)),
]

RETRY_PREPROCESS_METHODS = [
    "original",
    "enhanced",
    "lcd_sharp",
    "lcd_binary",
    "lcd_dark",
]

RESULT_TIME_STEP_MINUTES = 10
CHART_Y_MIN = 3.0
CHART_Y_MIN_TOP = 3.6
CHART_Y_MAX_LIMIT = 5.0
CHART_Y_MAJOR_STEP = 0.2
CHART_Y_MINOR_STEP = 0.1


# --- 辅助函数 ---

def load_config():
    """加载配置文件"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {"last_folder": "", "output_folder": ""}


def save_config(config):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def make_result(result, methods=None, best_method="", status=OCR_STATUS_DIRECT, note="", score=0.0):
    return {
        "result": str(result),
        "methods": methods or [],
        "best_method": best_method,
        "status": status,
        "note": note,
        "score": score
    }


def get_result_value(value):
    if isinstance(value, dict):
        return str(value.get("result", "0"))
    return str(value)


def get_result_status(value):
    if isinstance(value, dict):
        return value.get("status", OCR_STATUS_DIRECT)
    return OCR_STATUS_DIRECT


def get_result_note(value):
    if isinstance(value, dict):
        return value.get("note", "")
    return ""


def get_result_best_method(value):
    if isinstance(value, dict):
        return value.get("best_method", "")
    return ""


def read_image_cv2(path):
    """解决 Windows 下 opencv 不支持中文路径的问题"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None


def get_output_folder(output_folder):
    """未选择结果目录时，默认保存到程序当前运行目录。"""
    output_folder = (output_folder or "").strip()
    if not output_folder:
        output_folder = os.getcwd()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def save_results_to_txt(results, output_folder=None, timestamp=None):
    """将识别结果保存到txt文件"""
    try:
        output_folder = get_output_folder(output_folder)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_folder, f"{timestamp}_results.txt")
        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            for image_file, info in results.items():
                f.write(f"{image_file}:\n")
                f.write(f"{get_result_value(info)}\n")
                status = get_result_status(info)
                note = get_result_note(info)
                best_method = get_result_best_method(info)
                if status != OCR_STATUS_DIRECT:
                    f.write(f"标记: {status}\n")
                if best_method:
                    f.write(f"最佳方法: {best_method}\n")
                if note:
                    f.write(f"说明: {note}\n")
                f.write("\n")
        return filename
    except Exception as e:
        raise Exception(f"保存结果文件失败: {str(e)}")


def build_result_rows(results):
    rows = []
    for index, (img_name, info) in enumerate(results.items()):
        value = get_result_value(info)
        try:
            num_val = float(value)
        except:
            num_val = value
        rows.append({
            "图片名称": img_name,
            "时间(分钟)": index * RESULT_TIME_STEP_MINUTES,
            "识别结果": num_val,
            "标记": get_result_status(info),
            "最佳方法": get_result_best_method(info),
            "说明": get_result_note(info)
        })
    return rows


def save_results_to_excel(results, output_folder=None, timestamp=None):
    """将识别结果保存为 Excel 文件"""
    try:
        output_folder = get_output_folder(output_folder)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_folder, f"{timestamp}_results.xlsx")
        df = pd.DataFrame(build_result_rows(results))
        df.to_excel(filename, index=False)
        return filename
    except Exception as e:
        print(f"保存 Excel 失败: {str(e)}")
        return None


def get_chart_value(info):
    if get_result_status(info) == OCR_STATUS_FAILED:
        return np.nan
    try:
        return float(get_result_value(info))
    except:
        return np.nan


def get_nice_chart_step(max_value):
    for step in [10, 20, 50, 100, 200, 500, 1000]:
        if max_value <= step * 4:
            return step
    return 2000


def get_chart_y_max(y_values):
    valid_values = [value for value in y_values if not np.isnan(value)]
    if not valid_values:
        return CHART_Y_MIN_TOP
    data_max = max(valid_values)
    padded_max = data_max + CHART_Y_MINOR_STEP / 2
    rounded_max = np.ceil(padded_max / CHART_Y_MINOR_STEP) * CHART_Y_MINOR_STEP
    return min(max(CHART_Y_MIN_TOP, rounded_max), CHART_Y_MAX_LIMIT)


def save_results_chart(results, output_folder=None, timestamp=None):
    """按图片顺序生成识别数值随时间变化的折线图。"""
    try:
        output_folder = get_output_folder(output_folder)
        timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(output_folder, f"{timestamp}_results_chart.png")

        x_values = []
        y_values = []
        for index, info in enumerate(results.values()):
            x_values.append(index)
            y_values.append(get_chart_value(info))

        if not x_values or np.all(np.isnan(y_values)):
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter, MultipleLocator

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig_width = min(max(10, len(x_values) * 0.025), 16)
        fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=160)
        ax.plot(x_values, y_values, color="red", linewidth=2.2)
        ax.set_xlabel("时间(10min)")
        ax.set_ylabel("压强 (MPa)")

        max_time_index = max(x_values) if x_values else 1
        if max_time_index == 0:
            max_time_index = 1
        major_step = get_nice_chart_step(max_time_index)
        y_axis_max = get_chart_y_max(y_values)
        ax.set_xlim(0, max_time_index)
        ax.set_ylim(CHART_Y_MIN, y_axis_max)
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(MultipleLocator(max(1, major_step / 2)))
        ax.set_yticks(np.arange(CHART_Y_MIN, y_axis_max + 0.001, CHART_Y_MAJOR_STEP))
        ax.set_yticks(np.arange(CHART_Y_MIN, y_axis_max + 0.001, CHART_Y_MINOR_STEP), minor=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"保存趋势图失败: {str(e)}")
        return None


# --- 核心算法部分 ---

def init_paddleocr():
    """【修复】初始化 PaddleOCR，包含针对性参数调整"""
    try:
        from paddleocr import PaddleOCR
        # 显式定义 shape，确保它是字符串格式
        # 636 宽度给 AI 提供广角视野
        my_shape = "3, 48, 636"

        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=True,
            gpu_mem=1000,

            # --- 核心参数微调 ---
            det_db_thresh=0.05,  # 保持极低阈值，捕捉淡色笔画
            det_db_box_thresh=0.2,  # 保持低框阈值

            # 【关键修改】设定为 1.9
            # 1.6 太紧容易丢最后一位，2.2 太松容易粘连，1.9 是黄金平衡点
            det_db_unclip_ratio=1.9,

            # 【关键修改】直接传参
            rec_image_shape=my_shape
        )
        print(f"PaddleOCR 初始化成功，当前分辨率参数: {my_shape}")
        return ocr
    except Exception as e:
        print(f"PaddleOCR初始化失败: {str(e)}")
        return None


def preprocess_image_memory(img_array, method='enhanced'):
    """【最终修正版】降低分辨率防止压缩糊图，纯垂直加固"""
    if img_array is None:
        return None

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # 1. 【关键修改】设定目标大小为 320
    # 不要放大到 1000，否则 OCR 内部压缩会导致模糊。320 是最适合 PaddleOCR 的尺寸。
    height, width = gray.shape
    target_size = 320
    if height < target_size or width < target_size:
        scale = max(target_size / height, target_size / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    processed = gray

    # 2. 策略分支
    if method == 'thicken_lines':
        # 确保黑底白字
        if np.mean(gray) > 127:
            gray_inv = cv2.bitwise_not(gray)
        else:
            gray_inv = gray

        # 【关键修改】只做垂直膨胀，核宽度为 1
        # (4, 1) 表示垂直膨胀4像素，水平膨胀1像素（即不横向变粗）
        # 这能让断裂的“1”接上，但绝不会和旁边的“1”粘连
        kernel_v = np.ones((4, 1), np.uint8)
        processed = cv2.dilate(gray_inv, kernel_v, iterations=1)

        # 变回白底黑字
        processed = cv2.bitwise_not(processed)

    elif method == 'binary':
        # 配合 320px 的图，邻域使用 25 比较合适
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 2
        )
        processed = cv2.bitwise_not(processed)

    elif method == 'invert':
        processed = cv2.bitwise_not(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    elif method == 'enhanced':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)

    elif method == 'lcd_sharp':
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        processed = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    elif method == 'lcd_binary':
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        processed = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )

    elif method == 'lcd_dark':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        boosted = clahe.apply(gray)
        _, mask = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_v = np.ones((2, 1), np.uint8)
        mask = cv2.dilate(mask, kernel_v, iterations=1)
        processed = cv2.bitwise_not(mask)

    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)


def crop_image_region(img_array, region):
    """按比例裁剪仪表数字显示区域，避免品牌、单位、条形刻度干扰 OCR。"""
    height, width = img_array.shape[:2]
    left, top, right, bottom = region
    x1 = max(0, min(width - 1, int(width * left)))
    y1 = max(0, min(height - 1, int(height * top)))
    x2 = max(x1 + 1, min(width, int(width * right)))
    y2 = max(y1 + 1, min(height, int(height * bottom)))
    return img_array[y1:y2, x1:x2]


def score_ocr_candidate(text, score):
    final_score = score
    if '.' in text:
        final_score += 100
        final_score += len(text) * 10
    else:
        final_score += len(text) * 0.5
    return final_score


def extract_numbers(text):
    """【智能清洗版】自动剔除前缀噪点，修正异常大的数值"""
    if not text:
        return ""

    # 1. 基础清洗
    text = re.sub(r'[,;:_oO]', '.', text)
    text = text.replace(' ', '')

    # 2. 正则提取标准浮点数
    match = re.search(r'(\d+\.\d+)', text)

    if match:
        num_str = match.group(1)
        try:
            val = float(num_str)

            # =========== 【核心修改：范围过滤器】 ===========
            # 如果数值 > 10，说明前面识别到了边框或噪点（您的仪表读数应该是个位数）
            # 例如：OCR 识别出 "163.214" (把边框认成了16)
            # 我们只保留整数部分的最后一位
            if val > 10:
                if '.' in num_str:
                    int_part, dec_part = num_str.split('.')
                    # 只要整数部分的最后一位，加上原来的小数部分
                    # "163" -> "3"
                    if int_part:
                        new_int = int_part[-1]
                        num_str = f"{new_int}.{dec_part}"
            # ===========================================

            return num_str
        except:
            pass

    # 3. 兜底逻辑：处理连在一起的纯数字
    digits = re.findall(r'\d+', text)
    if digits:
        full_str = "".join(digits)
        # 如果是 4 位数 (3214) -> 3.214
        if len(full_str) >= 4 and '.' not in full_str:
            return full_str[0] + '.' + full_str[1:]
        # 如果是 3 位数 (321) -> 3.21
        elif len(full_str) == 3 and '.' not in full_str:
            return full_str[0] + '.' + full_str[1:]

    return ""


def ocr_execute_memory(img_array, ocr):
    """底层执行函数，处理内存图片并返回"""
    try:
        result = ocr.ocr(img_array, cls=False)
        if not result or not result[0]:
            return None, 0.0

        texts = []
        score_sum = 0
        count = 0

        for line in result[0]:
            if line and len(line) >= 2 and line[1]:
                text_content = line[1][0]
                confidence = line[1][1]
                texts.append(str(text_content))
                score_sum += float(confidence)
                count += 1

        combined_text = " ".join(texts)
        print(f"DEBUG - 原始识别内容: [{combined_text}]")

        final_number = extract_numbers(combined_text)
        if not final_number:
            return None, 0.0

        avg_score = score_sum / count if count > 0 else 0
        return final_number, avg_score

    except Exception as e:
        print(f"OCR执行出错: {e}")
        return None, 0.0


def ocr_image_with_stats(image_path, ocr, use_preprocessing=True):
    """主识别函数：并在内存中尝试多种方法，选取置信度最高的"""
    original_img = read_image_cv2(image_path)
    if original_img is None:
        return make_result("0", methods=[], best_method="error", status=OCR_STATUS_FAILED, note="图片读取失败")

    methods_to_try = ['original']
    if use_preprocessing:
        # 尝试多种预处理组合
        methods_to_try.extend(['thicken_lines', 'binary', 'invert', 'enhanced'])

    candidates = []
    used_methods = []

    for method in methods_to_try:
        try:
            if method == 'original':
                processed_img = original_img
            else:
                processed_img = preprocess_image_memory(original_img, method)

            used_methods.append(method)
            text, score = ocr_execute_memory(processed_img, ocr)

            if text:
                candidates.append({
                    "method": method,
                    "text": text,
                    "score": score_ocr_candidate(text, score)
                })
        except Exception:
            continue

    if candidates:
        # 按分数从高到低排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        return make_result(
            best["text"],
            methods=used_methods,
            best_method=best["method"],
            status=OCR_STATUS_DIRECT,
            score=best["score"]
        )

    retry_candidates = []
    for region_name, region in DIGIT_CROP_REGIONS:
        cropped_img = crop_image_region(original_img, region)
        for method in RETRY_PREPROCESS_METHODS:
            try:
                if method == 'original':
                    processed_img = cropped_img
                else:
                    processed_img = preprocess_image_memory(cropped_img, method)

                retry_method = f"{region_name}+{method}"
                used_methods.append(retry_method)
                text, score = ocr_execute_memory(processed_img, ocr)
                if text:
                    retry_candidates.append({
                        "method": retry_method,
                        "text": text,
                        "score": score_ocr_candidate(text, score)
                    })
            except Exception:
                continue

    if retry_candidates:
        retry_candidates.sort(key=lambda x: x["score"], reverse=True)
        best = retry_candidates[0]
        return make_result(
            best["text"],
            methods=used_methods,
            best_method=best["method"],
            status=OCR_STATUS_RETRY,
            note="原图识别失败后，通过裁剪数字区域二次识别恢复",
            score=best["score"]
        )

    return make_result("0", methods=used_methods, best_method="failed", status=OCR_STATUS_FAILED)


def parse_float_result(info):
    try:
        value = get_result_value(info)
        if value == "0":
            return None
        return float(value)
    except:
        return None


def decimal_places(value_text):
    value_text = str(value_text)
    if "." not in value_text:
        return 0
    return len(value_text.split(".", 1)[1])


def apply_interpolation_fill(results, ordered_files):
    """对连续失败段做线性补值，并保留“补值”标记，便于 Excel 复核。"""
    filled_count = 0
    index = 0
    total = len(ordered_files)

    while index < total:
        image_file = ordered_files[index]
        if image_file not in results or get_result_value(results[image_file]) != "0":
            index += 1
            continue

        block_start = index
        while index < total and ordered_files[index] in results and get_result_value(results[ordered_files[index]]) == "0":
            index += 1
        block_end = index - 1

        prev_index = block_start - 1
        next_index = index
        if prev_index < 0 or next_index >= total:
            continue

        prev_file = ordered_files[prev_index]
        next_file = ordered_files[next_index]
        prev_value = parse_float_result(results.get(prev_file))
        next_value = parse_float_result(results.get(next_file))
        if prev_value is None or next_value is None:
            continue

        block_size = block_end - block_start + 1
        precision = max(
            decimal_places(get_result_value(results[prev_file])),
            decimal_places(get_result_value(results[next_file])),
            3
        )

        for offset, fill_index in enumerate(range(block_start, block_end + 1), start=1):
            ratio = offset / (block_size + 1)
            filled_value = prev_value + (next_value - prev_value) * ratio
            filled_text = f"{filled_value:.{precision}f}"
            current_file = ordered_files[fill_index]
            current_info = results.get(current_file, {})
            current_methods = current_info.get("methods", []) if isinstance(current_info, dict) else []
            results[current_file] = make_result(
                filled_text,
                methods=current_methods,
                best_method="linear_interpolation",
                status=OCR_STATUS_FILLED,
                note=f"原图和二次识别均失败，按前后读数补值：{prev_file}={prev_value}，{next_file}={next_value}"
            )
            filled_count += 1

    return filled_count


# --- GUI 界面类 ---

class OCRGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("数字识别工具 (最终修正版)")
        self.root.geometry("700x600")
        self.ocr = None
        self.is_running = False
        self.config = load_config()
        self.stats = {
            "total_images": 0,
            "successful_recognitions": 0,
            "methods": {}
        }
        self.create_widgets()

    def create_widgets(self):
        # 文件夹选择区域
        folder_frame = ttk.LabelFrame(self.root, text="选择图片文件夹", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)

        self.folder_path = tk.StringVar(value=self.config.get("last_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).pack(side="left", padx=5)
        ttk.Button(folder_frame, text="选择文件夹", command=self.select_folder).pack(side="left")

        # 结果保存文件夹
        output_frame = ttk.LabelFrame(self.root, text="选择结果保存文件夹", padding=10)
        output_frame.pack(fill="x", padx=10, pady=5)

        self.output_path = tk.StringVar(value=self.config.get("output_folder", ""))
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side="left", padx=5)
        ttk.Button(output_frame, text="选择文件夹", command=self.select_output_folder).pack(side="left")

        # 控制按钮区域
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        self.start_btn = ttk.Button(button_frame, text="开始识别", command=self.start_ocr)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(button_frame, text="停止识别", command=self.stop_ocr, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        # 进度条
        progress_frame = ttk.LabelFrame(self.root, text="处理进度", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x")

        self.progress_label = ttk.Label(progress_frame, text="准备就绪")
        self.progress_label.pack(pady=5)

        # 日志显示区域
        log_frame = ttk.LabelFrame(self.root, text="处理日志", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)

        self.status_label = ttk.Label(self.root, text="就绪")
        self.status_label.pack(pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            self.log_message(f"已选择文件夹: {folder}")

    def select_output_folder(self):
        initial_dir = self.output_path.get() or self.folder_path.get() or os.getcwd()
        folder = filedialog.askdirectory(initialdir=initial_dir)
        if folder:
            self.output_path.set(folder)
            self.log_message(f"已选择结果保存文件夹: {folder}")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_ocr(self):
        if not self.folder_path.get():
            self.log_message("请先选择图片文件夹！")
            return
        if not os.path.exists(self.folder_path.get()):
            self.log_message(f"文件夹不存在: {self.folder_path.get()}")
            return
        if self.output_path.get() and not os.path.exists(self.output_path.get()):
            self.log_message(f"结果保存文件夹不存在: {self.output_path.get()}")
            return

        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.progress_label.config(text="初始化中...")
        self.status_label.config(text="运行中")

        thread = threading.Thread(target=self.run_ocr_task)
        thread.daemon = True
        thread.start()

    def stop_ocr(self):
        self.is_running = False
        self.log_message("正在停止识别...")
        self.status_label.config(text="正在停止")

    def run_ocr_task(self):
        try:
            self.log_message("正在初始化识别引擎...")
            self.ocr = init_paddleocr()
            if self.ocr is None:
                self.log_message("引擎初始化失败！")
                self.reset_ui()
                return

            img_folder = self.folder_path.get()
            output_folder = get_output_folder(self.output_path.get())
            self.log_message(f"结果将保存到: {output_folder}")
            image_files = sorted([f for f in os.listdir(img_folder)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

            if not image_files:
                self.log_message("未找到图片文件！")
                self.reset_ui()
                return

            self.log_message(f"找到 {len(image_files)} 个图片文件")
            results = {}
            processed_files = []
            total_files = len(image_files)

            for i, image_file in enumerate(image_files):
                if not self.is_running:
                    self.log_message("识别已停止")
                    break

                self.progress_var.set((i / total_files) * 100)
                self.progress_label.config(text=f"正在处理: {image_file} ({i + 1}/{total_files})")
                image_path = os.path.join(img_folder, image_file)
                self.log_message(f"处理图片: {image_file}")

                try:
                    result_info = ocr_image_with_stats(image_path, self.ocr, use_preprocessing=True)
                    numbers = get_result_value(result_info)
                    results[image_file] = result_info
                    processed_files.append(image_file)

                    self.stats["total_images"] += 1
                    if numbers != "0":
                        self.stats["successful_recognitions"] += 1

                    status = get_result_status(result_info)
                    if status == OCR_STATUS_RETRY:
                        self.log_message(f"识别结果: {numbers} ({status})")
                    else:
                        self.log_message(f"识别结果: {numbers}")
                except Exception as e:
                    self.log_message(f"处理失败: {str(e)}")
                    results[image_file] = make_result("0", best_method="exception", status=OCR_STATUS_FAILED, note=str(e))
                    processed_files.append(image_file)
                    self.stats["total_images"] += 1

                self.progress_var.set(((i + 1) / total_files) * 100)

            if self.is_running and results:
                filled_count = apply_interpolation_fill(results, processed_files)
                if filled_count:
                    self.log_message(f"已补值 {filled_count} 张失败图片，Excel 中标记为“{OCR_STATUS_FILLED}”")

                result_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                txt_filename = save_results_to_txt(results, output_folder, result_timestamp)
                self.log_message(f"已保存 TXT: {txt_filename}")

                xls_filename = save_results_to_excel(results, output_folder, result_timestamp)
                if xls_filename:
                    self.log_message(f"已保存 Excel: {xls_filename}")

                chart_filename = save_results_chart(results, output_folder, result_timestamp)
                if chart_filename:
                    self.log_message(f"已保存趋势图: {chart_filename}")
                else:
                    self.log_message("趋势图未生成：没有可绘制的有效数值")

                self.log_message("所有结果保存完成！")

                total_images = len(results)
                successful_recognitions = sum(1 for info in results.values() if get_result_value(info) != "0")
                retry_count = sum(1 for info in results.values() if get_result_status(info) == OCR_STATUS_RETRY)
                filled_count = sum(1 for info in results.values() if get_result_status(info) == OCR_STATUS_FILLED)
                success_rate = (successful_recognitions / total_images * 100) if total_images > 0 else 0
                self.log_message(
                    f"识别统计: 共处理 {total_images} 张图片，有效结果 {successful_recognitions} 张，"
                    f"二次识别 {retry_count} 张，补值 {filled_count} 张，最终有效率 {success_rate:.1f}%")

                self.config["last_folder"] = img_folder
                self.config["output_folder"] = output_folder
                save_config(self.config)

            self.progress_label.config(text="处理完成")
            self.status_label.config(text="完成")

        except Exception as e:
            self.log_message(f"发生错误: {str(e)}")
            self.status_label.config(text="错误")
        finally:
            self.reset_ui()

    def reset_ui(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.is_running = False

    def run(self):
        self.root.mainloop()


def main():
    gui = OCRGUI()
    gui.run()


if __name__ == "__main__":
    main()
