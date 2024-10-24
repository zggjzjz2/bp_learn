import pyautogui
import time

file_path = r'C:\Users\dell\Desktop\BPProject1\SVD_learn.py'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    print("请在5秒内切换到答题界面...")
    time.sleep(5)

    for line in lines:
        pyautogui.typewrite(line)
        pyautogui.press('enter')
    print("代码输入完成。")

except FileNotFoundError:
    print(f"文件未找到，请检查路径是否正确：{file_path}")

except Exception as e:
    print(f"发生错误: {e}")