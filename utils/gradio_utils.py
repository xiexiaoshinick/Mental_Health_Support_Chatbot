from __future__ import annotations
import base64
import html
import os
import re
from urllib import parse

import json
import markdown
from gradio.components import Chatbot as ChatBotBase

# 图片本地路径转换为 base64 格式
def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split('.')[-1]
    if ext not in ['gif', 'jpeg', 'png']:
        ext = 'jpeg'

    with open(image_path, 'rb') as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode('utf-8')

        # 生成base64编码的地址
        base64_url = f'data:image/{ext};base64,{base64_data}'
        return base64_url


def convert_url(text, new_filename):
    # Define the pattern to search for
    # This pattern captures the text inside the square brackets, the path, and the filename
    pattern = r'!\[([^\]]+)\]\(([^)]+)\)'

    # Define the replacement pattern
    # \1 is a backreference to the text captured by the first group ([^\]]+)
    replacement = rf'![\1]({new_filename})'

    # Replace the pattern in the text with the replacement
    return re.sub(pattern, replacement, text)


def format_cover_html(model_chinese_name,description, bot_avatar_path):
    if bot_avatar_path:
        image_src = covert_image_to_base64(bot_avatar_path)
    else:
        image_src = '//img.alicdn.com/imgextra/i3/O1CN01YPqZFO1YNZerQfSBk_!!6000000003047-0-tps-225-225.jpg'
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} />
    </div>
    <div class="bot_name">{model_chinese_name}</div>
    <div class="bot_desp">{description}</div>
</div>
"""
