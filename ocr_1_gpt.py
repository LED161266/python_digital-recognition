import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import os
import sys
import re
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
from datetime import datetime
CONFIG_FILE = "ocr_config.json"
import os
MODEL_HOME = r"D:\paddle_models"
os.makedirs(MODEL_HOME, exist_ok=True)

# 让 PaddleOCR/PaddleX/Paddle 的缓存都去英文路径
os.environ["PADDLEOCR_HOME"] = MODEL_HOME
os.environ["PADDLE_HOME"] = MODEL_HOME
os.environ["PADDLEX_HOME"] = MODEL_HOME

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

class OCRGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("数字识别工具")
        self.root.geometry("700x600")

        # OCR实例
        self.ocr = None
        self.is_running = False

        # 加载配置
        self.config = load_config()

        # 统计数据
        self.stats = {
            "total_images": 0,
            "successful_recognitions": 0,
            "methods": {}  # 每种方法的统计
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

        # 状态栏
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

        # 在新线程中运行OCR
        thread = threading.Thread(target=self.run_ocr_task)
        thread.daemon = True
        thread.start()

    def stop_ocr(self):
        self.is_running = False
        self.log_message("正在停止识别...")
        self.status_label.config(text="正在停止")

    def run_ocr_task(self):
        try:
            # 初始化OCR
            self.log_message("正在初始化识别引擎...")
            self.ocr = init_paddleocr()
            if self.ocr is None:
                self.log_message("引擎初始化失败！")
                self.reset_ui()
                return

            # 获取图片文件列表
            img_folder = self.folder_path.get()
            image_files = [f for f in os.listdir(img_folder)
                          if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if not image_files:
                self.log_message("未找到图片文件！")
                self.reset_ui()
                return

            self.log_message(f"找到 {len(image_files)} 个图片文件")

            # 处理图片
            results = {}
            total_files = len(image_files)

            for i, image_file in enumerate(image_files):
                if not self.is_running:
                    self.log_message("识别已停止")
                    break

                self.progress_var.set((i / total_files) * 100)
                self.progress_label.config(text=f"正在处理: {image_file} ({i+1}/{total_files})")

                image_path = os.path.join(img_folder, image_file)
                self.log_message(f"处理图片: {image_file}")

                try:
                    # 调用ocr_image并获取详细信息
                    result_info = ocr_image_with_stats(image_path, self.ocr, use_preprocessing=True)
                    numbers = result_info["result"]
                    used_methods = result_info["methods"]

                    results[image_file] = numbers

                    # 更新统计数据
                    self.stats["total_images"] += 1
                    if numbers != "0":
                        self.stats["successful_recognitions"] += 1

                    # 更新方法统计
                    for method in used_methods:
                        if method not in self.stats["methods"]:
                            self.stats["methods"][method] = {"total": 0, "success": 0, "rate": 0.0}
                        self.stats["methods"][method]["total"] += 1
                        if numbers != "0":
                            self.stats["methods"][method]["success"] += 1
                            # 重新计算成功率
                            total = self.stats["methods"][method]["total"]
                            success = self.stats["methods"][method]["success"]
                            self.stats["methods"][method]["rate"] = round(success / total * 100, 1)

                    self.log_message(f"识别结果: {numbers}")
                except Exception as e:
                    error_msg = str(e)
                    self.log_message(f"处理失败: {error_msg}")
                    results[image_file] = "0"
                    self.stats["total_images"] += 1

                # 更新进度
                self.progress_var.set(((i + 1) / total_files) * 100)

            # 保存结果
            if self.is_running and results:
                filename = save_results_to_txt(results)
                self.log_message(f"正在保存结果到: {filename}")
                self.log_message("结果保存完成！")

                # 统计识别成功率
                total_images = len(results)
                successful_recognitions = sum(1 for numbers in results.values() if numbers != "0")
                success_rate = (successful_recognitions / total_images * 100) if total_images > 0 else 0

                self.log_message(f"识别统计: 共处理 {total_images} 张图片，成功识别 {successful_recognitions} 张，成功率 {success_rate:.1f}%")

                # 预处理方法统计报告（已注释掉）
                # self.log_message("预处理方法统计:")
                # methods = ['original', 'enhanced', 'binary', 'denoise', 'resize']
                # for method in methods:
                #     if method in self.stats["methods"]:
                #         method_stats = self.stats["methods"][method]
                #         total_used = method_stats["total"]
                #         successful = method_stats["success"]
                #         rate = method_stats["rate"]
                #         self.log_message(f"  {method}: 使用{total_used}次，成功{successful}次，成功率{rate:.1f}%")

                # 保存文件夹路径到配置
                self.config["last_folder"] = img_folder
                save_config(self.config)
                self.log_message("文件夹路径已保存")

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

def save_results_to_txt(results):
    """将识别结果保存到按时间命名的txt文件中"""
    try:
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_results.txt"

        # 写入文件
        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            for image_file, numbers in results.items():
                f.write(f"{image_file}:\n")
                f.write(f"{numbers}\n")
                f.write("\n")  # 空行分隔

        return filename

    except Exception as e:
        raise Exception(f"保存结果文件失败: {str(e)}")

def preprocess_image(image_path, method='enhanced'):
    """预处理图片以提高数字识别精度

    Args:
        image_path: 图片路径
        method: 预处理方法 ('enhanced', 'binary', 'denoise', 'resize')
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return image_path  # 返回原图

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if method == 'enhanced':
            # 方法1: 对比度增强 + 锐化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(enhanced, -1, kernel)

        elif method == 'binary':
            # 方法2: 自适应二值化
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'denoise':
            # 方法3: 去噪 + 增强
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            processed = clahe.apply(denoised)

        elif method == 'resize':
            # 方法4: 放大图片 + 增强（提高小数字的识别率）
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale = max(1000 / height, 1000 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                resized = gray
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(resized)

        else:
            processed = gray

        # 保存临时处理后的图片
        base_name = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1]
        temp_path = f"{base_name}_processed_{method}{ext}"
        cv2.imwrite(temp_path, processed)

        return temp_path

    except Exception as e:
        print(f"图片预处理失败: {str(e)}")
        return image_path  # 返回原图路径

def preprocess_multiple(image_path):
    """对图片进行多种预处理，返回多个处理后的路径"""
    # methods = ['enhanced', 'binary', 'denoise', 'resize']
    methods = ['enhanced', 'denoise']
    processed_paths = []

    for method in methods:
        try:
            processed = preprocess_image(image_path, method)
            if processed != image_path:
                processed_paths.append(processed)
        except:
            continue

    return processed_paths

def validate_number(num_str):
    """验证数字格式是否合理"""
    if not num_str:
        return False

    # 检查是否包含小数点
    if '.' not in num_str:
        return False

    # 检查格式：应该是 x.xxx 或 x.xx 等格式
    if not re.match(r'^\d+\.\d+$', num_str):
        return False

    # 检查小数位数（通常3-4位）
    parts = num_str.split('.')
    if len(parts) != 2:
        return False

    # 整数部分通常是1-2位，小数部分通常是2-4位
    int_part, dec_part = parts
    if len(int_part) > 3 or len(dec_part) < 2 or len(dec_part) > 5:
        return False

    return True

def extract_numbers(text):
    """从识别结果中提取数字，包括小数点"""
    if not text:
        return ""

    # 首先尝试识别包含标准小数点的数字
    standard_pattern = r'\d+\.?\d*|\.\d+'
    standard_numbers = re.findall(standard_pattern, text)

    # 如果没找到标准小数点，尝试识别可能的变体
    # 小数点可能被识别为其他符号：, ; : - _ 等
    if not standard_numbers:
        # 查找可能的数字+分隔符+数字模式
        variant_patterns = [
            r'\d+[,;:_-]\d+',  # 3,432, 3;432, 3:432, 3-432, 3_432
            r'\d+\s*[.,;:_-]\s*\d+',  # 3 . 432, 3 , 432 等
            r'\d+\s+\.\s+\d+',  # 3 . 432 (带空格)
            r'\d+[oO]\d+',  # 3o432 (O被误识别为小数点)
        ]

        for pattern in variant_patterns:
            candidates = re.findall(pattern, text)
            if candidates:
                # 转换为标准小数点格式
                converted = []
                for candidate in candidates:
                    # 将各种分隔符替换为小数点
                    converted_num = re.sub(r'[,;:_-oO]', '.', candidate)
                    # 移除多余空格
                    converted_num = re.sub(r'\s+', '', converted_num)
                    # 验证格式：应该是 x.xxx 格式
                    if re.match(r'^\d+\.\d+$', converted_num):
                        converted.append(converted_num)
                standard_numbers.extend(converted)

    # 只保留包含小数点的数字，舍弃纯数字
    numbers = standard_numbers

    # 清理和验证找到的数字，只保留包含小数点的数字
    valid_numbers = []
    for num in numbers:
        # 移除前导和尾随的点号
        num = num.strip('.')
        if num and '.' in num:  # 只保留包含小数点的数字
            # 确保不是只有点号
            if num != '.':
                # 标准化格式
                if num.startswith('.'):
                    num = '0' + num
                if num.endswith('.'):
                    num = num.rstrip('.')
                # 验证数字格式是否合理
                if validate_number(num):
                    valid_numbers.append(num)

    # 合并相邻的数字片段（处理OCR可能分割的情况）
    if valid_numbers:
        merged_numbers = []
        current = valid_numbers[0]

        for num in valid_numbers[1:]:
            # 更保守的合并策略：只有当明确是分割的小数点数字时才合并
            # 例如：["3", ".432"] -> ["3.432"]，但不会合并 ["3", "479"]
            if current.isdigit() and num.startswith('.') and len(num) > 1 and len(current) <= 2:
                # 合并小数点的情况，如 "3" + ".432" -> "3.432"
                current += num
            elif current.endswith('.') and num.isdigit() and len(current) <= 3:
                # 合并小数点被分割的情况，如 "3." + "432" -> "3.432"
                current += num
            else:
                # 不合并，直接添加
                merged_numbers.append(current)
                current = num

        merged_numbers.append(current)

        return ' '.join(merged_numbers)

    return ""

def init_paddleocr():
    """初始化PaddleOCR - GPU数字识别（稳定英文路径缓存）"""
    try:
        import os

        model_home = r"D:\paddle_models"
        os.makedirs(model_home, exist_ok=True)

        # 保存旧环境变量（很重要，初始化后恢复，避免影响文件对话框）
        old_userprofile = os.environ.get("USERPROFILE")
        old_home = os.environ.get("HOME")

        # ✅ PaddleOCR 在 Windows 上通常用 USERPROFILE 作为 ~ 的来源
        os.environ["USERPROFILE"] = model_home
        os.environ["HOME"] = model_home

        # 这几个留着也没坏处
        os.environ["PADDLEOCR_HOME"] = model_home
        os.environ["PADDLE_HOME"] = model_home
        os.environ["PADDLEX_HOME"] = model_home

        print("USERPROFILE =", os.environ.get("USERPROFILE"))
        print("HOME =", os.environ.get("HOME"))

        from paddleocr import PaddleOCR
        import paddle
        paddle.set_device("gpu")

        ocr = PaddleOCR(
            lang='en',
            use_gpu=True,
            gpu_mem=500,
            use_angle_cls=False
        )

        print("PaddleOCR GPU 初始化成功")
        return ocr

    except Exception as e:
        print(f"PaddleOCR初始化失败: {str(e)}")
        return None

    finally:
        # ✅ 初始化完恢复（避免影响 tkinter 文件选择框的 Desktop 路径）
        if old_userprofile is not None:
            os.environ["USERPROFILE"] = old_userprofile
        else:
            os.environ.pop("USERPROFILE", None)

        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            os.environ.pop("HOME", None)




def ocr_image_single(image_path, ocr):
    """对单张图片进行OCR识别（内部函数）"""
    try:
        result = ocr.ocr(image_path)
        all_texts = []
        if result and len(result) > 0:
            result_dict = result[0]
            if 'rec_texts' in result_dict:
                texts = result_dict['rec_texts']
                scores = result_dict.get('rec_scores', [])

                for i, text in enumerate(texts):
                    if text and text.strip():
                        # 提高置信度阈值，过滤低质量识别
                        confidence_threshold = 0.3
                        if i < len(scores) and scores[i] > confidence_threshold:
                            all_texts.append(text)
                        elif not scores:
                            all_texts.append(text)

        combined_text = ' '.join(all_texts)
        numbers = extract_numbers(combined_text)
        return numbers if numbers else None
    except:
        return None

def ocr_image(image_path, ocr, use_preprocessing=True):
    """使用PaddleOCR对单张图片进行数字识别（多策略）"""
    try:
        results = []
        temp_files = []

        # 策略1: 尝试原图
        try:
            original_result = ocr_image_single(image_path, ocr)
            if original_result:
                results.append(original_result)
        except:
            pass

        # 策略2-5: 尝试多种预处理方法
        if use_preprocessing:
            processed_paths = preprocess_multiple(image_path)
            temp_files.extend(processed_paths)

            for processed_path in processed_paths:
                try:
                    processed_result = ocr_image_single(processed_path, ocr)
                    if processed_result:
                        results.append(processed_result)
                except:
                    continue

        # 选择最佳结果：优先选择包含小数点的数字
        best_result = None
        for result in results:
            if result and '.' in result:
                best_result = result
                break

        # 如果没有找到小数点数字，选择第一个非空结果
        if not best_result and results:
            best_result = results[0]

        # 清理临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        return best_result if best_result else "0"
    except Exception as e:
        raise Exception(f"PaddleOCR识别失败: {str(e)}")

def ocr_image_with_stats(image_path, ocr, use_preprocessing=True):
    """使用PaddleOCR对单张图片进行数字识别，返回详细信息"""
    try:
        results = []
        temp_files = []
        used_methods = ["original"]  # 总是包含原图

        # 策略1: 尝试原图
        try:
            original_result = ocr_image_single(image_path, ocr)
            if original_result:
                results.append(("original", original_result))
        except:
            pass

        # 策略2-5: 尝试多种预处理方法
        if use_preprocessing:
            processed_paths = preprocess_multiple(image_path)
            temp_files.extend(processed_paths)

            for processed_path in processed_paths:
                try:
                    processed_result = ocr_image_single(processed_path, ocr)
                    if processed_result:
                        # 从文件名中提取方法名
                        method_name = "unknown"
                        if "_processed_" in processed_path:
                            parts = processed_path.split("_processed_")
                            if len(parts) > 1:
                                method_name = parts[1].split(".")[0]
                        results.append((method_name, processed_result))
                        used_methods.append(method_name)
                except:
                    continue

        # 选择最佳结果：优先选择包含小数点的数字
        best_result = None
        best_method = "original"
        for method, result in results:
            if result and '.' in result:
                best_result = result
                best_method = method
                break

        # 如果没有找到小数点数字，选择第一个非空结果
        if not best_result and results:
            best_method, best_result = results[0]

        # 清理临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        return {
            "result": best_result if best_result else "0",
            "methods": used_methods,
            "best_method": best_method
        }
    except Exception as e:
        return {
            "result": "0",
            "methods": ["original"],
            "best_method": "original",
            "error": str(e)
        }

def main():
    # 创建并运行GUI
    gui = OCRGUI()
    gui.run()

if __name__ == "__main__":
    main()
