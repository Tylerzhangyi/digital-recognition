from flask import Flask, request, jsonify
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# 初始化 S3 和 SQS 客户端
s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION'))

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        data = request.json
        images = data.get('images', [])

        if not images:
            return jsonify({'error': '未提供图片数据'}), 400

        queue_url = os.getenv('SQS_QUEUE_URL')

        for image in images:
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=image['data'],
                MessageAttributes={
                    'ImageName': {'StringValue': image['name'], 'DataType': 'String'}
                }
            )

        return jsonify({'message': f'{len(images)} 张图片已成功上传并排队处理'}), 200

    except Exception as e:
        print(f"处理上传请求时出错: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/get_result_url', methods=['GET'])
def get_result_url():
    image_name = request.args.get('image_name')
    if not image_name:
        return jsonify({'error': 'Missing image_name parameter'}), 400

    bucket_name = os.getenv('S3_BUCKET_NAME')
    file_name = f'results/{image_name}.txt'

    try:
        # 生成预签名 URL
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600  # URL 有效期为 1 小时
        )
        return jsonify({'url': url})
    except Exception as e:
        print(f"生成预签名 URL 时出错: {e}")
        return jsonify({'error': '无法生成结果 URL'}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)