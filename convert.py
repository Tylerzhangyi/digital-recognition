import base64


def image_to_base64(image_path):
    """
    将本地图片转换为Base64编码字符串

    :param image_path: 图片的本地路径
    :return: Base64编码的字符串，如果发生错误则返回None
    """
    try:
        # 以二进制模式打开图片文件
        with open(image_path, "rb") as image_file:
            # 读取文件内容并进行Base64编码
            encoded_string = base64.b64encode(image_file.read())
            # 将字节类型转换为字符串类型
            return encoded_string.decode('utf-8')
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
    except Exception as e:
        print(f"发生错误：{e}")

    return None


# 示例调用
if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "/Users/zhangyi/Desktop/digit_identify/0.png"
    base64_string = image_to_base64(image_path)

    if base64_string:
        print("Base64编码的图片字符串:")
        print(base64_string)