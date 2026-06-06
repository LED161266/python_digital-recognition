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


# --- 辅助函数 ---

def load_config():
    """加载配置文件"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {"last_folder": ""}


def save_config(config):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def read_image_cv2(path):
    """解决 Windows 下 opencv 不支持中文路径的问题"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None


def save_results_to_txt(results):
    """将识别结果保存到txt文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_results.txt"
        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            for image_file, numbers in results.items():
                f.write(f"{image_file}:\n")
                f.write(f"{numbers}\n")
                f.write("\n")
        return filename
    except Exception as e:
        raise Exception(f"保存结果文件失败: {str(e)}")


def save_results_to_excel(results):
    """将识别结果保存为 Excel 文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_results.xlsx"
        data = []
        for img_name, value in results.items():
            try:
                num_val = float(value)
            except:
                num_val = value
            data.append({"图片名称": img_name, "识别结果": num_val})
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        return filename
    except Exception as e:
        print(f"保存 Excel 失败: {str(e)}")
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

    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)


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
        return {"result": "0", "methods": [], "best_method": "error"}

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
                final_score = score
                # 【评分逻辑】优先选择带小数点的，优先选择更长的
                if '.' in text:
                    final_score += 100
                    final_score += len(text) * 10
                else:
                    final_score += len(text) * 0.5

                candidates.append({
                    "method": method,
                    "text": text,
                    "score": final_score
                })
        except Exception:
            continue

    if not candidates:
        return {
            "result": "0",
            "methods": used_methods,
            "best_method": "failed"
        }

    # 按分数从高到低排序
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    return {
        "result": best["text"],
        "methods": used_methods,
        "best_method": best["method"]
    }


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
            image_files = [f for f in os.listdir(img_folder)
                           if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if not image_files:
                self.log_message("未找到图片文件！")
                self.reset_ui()
                return

            self.log_message(f"找到 {len(image_files)} 个图片文件")
            results = {}
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
                    numbers = result_info["result"]
                    results[image_file] = numbers

                    self.stats["total_images"] += 1
                    if numbers != "0":
                        self.stats["successful_recognitions"] += 1

                    self.log_message(f"识别结果: {numbers}")
                except Exception as e:
                    self.log_message(f"处理失败: {str(e)}")
                    results[image_file] = "0"
                    self.stats["total_images"] += 1

                self.progress_var.set(((i + 1) / total_files) * 100)

            if self.is_running and results:
                txt_filename = save_results_to_txt(results)
                self.log_message(f"已保存 TXT: {txt_filename}")

                xls_filename = save_results_to_excel(results)
                if xls_filename:
                    self.log_message(f"已保存 Excel: {xls_filename}")

                self.log_message("所有结果保存完成！")

                total_images = len(results)
                successful_recognitions = sum(1 for numbers in results.values() if numbers != "0")
                success_rate = (successful_recognitions / total_images * 100) if total_images > 0 else 0
                self.log_message(
                    f"识别统计: 共处理 {total_images} 张图片，成功识别 {successful_recognitions} 张，成功率 {success_rate:.1f}%")

                self.config["last_folder"] = img_folder
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