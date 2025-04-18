import boto3
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# 加载环境变量
load_dotenv()

# 初始化 SQS 客户端
def get_sqs_client():
    try:
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region_name = os.getenv('AWS_REGION')

        if not all([aws_access_key_id, aws_secret_access_key, region_name]):
            raise ValueError("缺少必要的环境变量: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")

        sqs = boto3.client(
            'sqs',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        return sqs
    except Exception as e:
        print(f"初始化 SQS 客户端失败: {e}")
        raise

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 SQS 客户端
sqs = get_sqs_client()
queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/961341521760/task.fifo'

# 新增 API 接口：接收 Base64 图像并推送到 SQS
@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        data = request.json
        images = data.get('images', [])

        if not images:
            return jsonify({'error': '未提供图片数据'}), 400

        for image in images:
            image_name = image.get('name')
            image_data = image.get('data')

            if not image_data:
                continue

            # 将 Base64 数据推送到 SQS 队列
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=image_data,
                MessageGroupId='image-processing'  # FIFO 队列需要 Group ID
            )
            print(f"已推送图片 {image_name} 到 SQS 队列")

        return jsonify({'message': f'{len(images)} 张图片已成功上传并排队处理'}), 200

    except Exception as e:
        print(f"处理上传请求时出错: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)