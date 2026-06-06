import pandas as pd


def txt_to_excel(txt_filename, excel_filename):
    """
    将包含文件名和数值的TXT文件转换为Excel文件。
    """
    data_list = []

    try:
        with open(txt_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_filename = None

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 判断逻辑：
            # 如果行尾是冒号 ':'，我们认为它是文件名
            if line.endswith(':'):
                current_filename = line[:-1]  # 去掉冒号
            # 如果不是文件名，且当前已经缓存了文件名，我们认为这是对应的数值
            elif current_filename is not None:
                try:
                    # 尝试转换成浮点数，方便在Excel里做计算
                    value = float(line)

                    data_list.append({
                        'Image Name': current_filename,
                        'Value': value
                    })

                    # 重置文件名，防止错位
                    current_filename = None
                except ValueError:
                    print(f"警告: 无法转换数值 '{line}'，已跳过。")

        # 使用pandas创建DataFrame
        if data_list:
            df = pd.DataFrame(data_list)

            # 保存为Excel
            df.to_excel(excel_filename, index=False)
            print(f"成功！文件已保存为: {excel_filename}")
            print(f"共处理了 {len(data_list)} 条数据。")
        else:
            print("未找到有效数据。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_filename}")


# --- 使用方法 ---
# 将这里的名字改成你实际的txt文件名
input_txt = '20251220180551_results.txt'
output_excel = 'output_results.xlsx'

if __name__ == '__main__':
    # 建议先创建一个空的同名txt文件进行测试，或者确保txt文件和脚本在同一目录下
    # 由于你已经上传了文件内容，你可以直接把内容复制到本地的txt文件中运行
    txt_to_excel(input_txt, output_excel)