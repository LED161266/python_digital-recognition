import cv2
import numpy as np
import easyocr
import glob
import os
import pandas as pd


# 解决 OpenCV 无法读取中文路径的问题
def imread_cn(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# 初始化 OCR
reader = easyocr.Reader(['en'])


def extract_screen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnts[0])
    screen = gray[y:y+h, x:x+w]
    return screen


def enhance(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.equalizeHist(img)
    return img


def ocr_digits(img):
    result = reader.readtext(img, detail=0)
    if not result:
        return None
    return result[0]


def process_one_image(path):
    img = imread_cn(path)   # ← 用中文路径安全读取方式
    if img is None:
        print("❌ 无法读取图像:", path)
        return None

    screen = extract_screen(img)
    screen = enhance(screen)
    text = ocr_digits(screen)
    return text


def batch_process_to_excel(folder, excel_path="result.xlsx"):
    image_list = glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.png"))

    if not image_list:
        print("❌ 文件夹内没有找到 JPG/PNG 图片")
        return

    # 自动补充 .xlsx 后缀
    if not excel_path.lower().endswith(".xlsx"):
        excel_path = excel_path + ".xlsx"

    print(f"开始识别，总共找到 {len(image_list)} 张图片...\n")

    records = []
    for path in image_list:
        text = process_one_image(path)
        print(f"{os.path.basename(path)}  →  {text}")
        records.append({
            "文件名": os.path.basename(path),
            "完整路径": os.path.abspath(path),
            "识别结果": text
        })

    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"\n✅ 已保存到 Excel：{excel_path}")


if __name__ == "__main__":
    target_folder = r"D:\qq\zp1"

    # 这里必须是一个文件，而不是目录
    output_excel = r"D:\read\result1.xlsx"

    batch_process_to_excel(target_folder, output_excel)

