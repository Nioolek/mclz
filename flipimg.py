from PIL import Image
import exifread

# 打开图像文件
image_path = "/data/gulingrui/code/mclz/data/picture/ele_98e94b49f53d45323d587097c7e9b135.jpg"
with open(image_path, 'rb') as f:
    tags = exifread.process_file(f)

# 输出所有标记及其值
for tag in tags.keys():
    print("{}: {}".format(tag, tags[tag]))