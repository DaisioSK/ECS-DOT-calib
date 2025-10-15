from PIL import Image, ImageOps
import numpy as np

# 假设你想要 28x28 灰度图
w, h = 28, 28  
# 生成一个手写 “8” 或者其它你想要的数字／图案
# 这里简单示例：一个白底黑字 “8” 用 PIL 画
img = Image.new("L", (w, h), 255)  # L 模式 = 灰度，255 白背景
# 你需要自己画数字 8（用 draw / ImageDraw）
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)
# 用一个系统字体（路径可能在 /usr/share/fonts 中找）：
f = ImageFont.load_default()
draw.text((4, 4), "8", font=f, fill=0)  # 黑色数字“8”

# 取反（invert）如果你要 “invert” 的效果
img_inv = ImageOps.invert(img)

# 保存为 PGM（P5）
img_inv.save("data/img/lenet/eight_invert.pgm", format="PPM")  # pillow 会自动写 PGM/PPM

# 或者显式写 PGM
# img_inv.save("eight_invert.pgm")
