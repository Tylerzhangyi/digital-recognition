import boto3
import os
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 初始化 AWS 客户端
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

def get_ec2_client():
    try:
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region_name = os.getenv('AWS_REGION')

        if not all([aws_access_key_id, aws_secret_access_key, region_name]):
            raise ValueError("缺少必要的环境变量: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")

        ec2 = boto3.client(
            'ec2',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        return ec2
    except Exception as e:
        print(f"初始化 EC2 客户端失败: {e}")
        raise

# 获取 SQS 队列中的消息数量
def get_queue_length(sqs, queue_url):
    try:
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        message_count = int(response['Attributes']['ApproximateNumberOfMessages'])
        return message_count
    except Exception as e:
        print(f"获取队列长度失败: {e}")
        return 0

# 启动 EC2 实例
def start_instances(ec2, instance_count, ami_id, instance_type='t2.micro'):
    try:
        instances = ec2.run_instances(
            ImageId=ami_id,
            MinCount=instance_count,
            MaxCount=instance_count,
            InstanceType=instance_type,
            KeyName=os.getenv('EC2_KEY_PAIR'),  # 替换为您的 EC2 密钥对名称
            SecurityGroupIds=[os.getenv('SECURITY_GROUP')],  # 替换为您的安全组 ID
            IamInstanceProfile={'Name': os.getenv('INSTANCE_PROFILE')}  # IAM 角色名称
        )
        instance_ids = [instance['InstanceId'] for instance in instances['Instances']]
        print(f"已启动 {len(instance_ids)} 个实例: {instance_ids}")
        return instance_ids
    except Exception as e:
        print(f"启动实例失败: {e}")
        return []

# 终止 EC2 实例
def terminate_instances(ec2, instance_ids):
    try:
        ec2.terminate_instances(InstanceIds=instance_ids)
        print(f"已终止实例: {instance_ids}")
    except Exception as e:
        print(f"终止实例失败: {e}")

# 主逻辑：弹性计算调度
def elastic_compute_scheduler():
    # 初始化客户端
    sqs = get_sqs_client()
    ec2 = get_ec2_client()

    # 配置参数
    queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/961341521760/task.fifo'
    ami_id = os.getenv('AMI_ID')  # 替换为您的 AMI ID
    max_instances = 10  # 最大实例数限制
    min_instances = 1   # 最小实例数限制
    instance_type = 't2.micro'

    # 当前运行的实例 ID 列表
    running_instance_ids = []

    while True:
        try:
            # 获取队列长度
            queue_length = get_queue_length(sqs, queue_url)
            print(f"当前队列中有 {queue_length} 条消息")

            # 计算需要的实例数量
            desired_instances = min(max(queue_length // 5 + 1, min_instances), max_instances)
            print(f"期望的实例数量: {desired_instances}")

            # 如果实例不足，启动新实例
            if len(running_instance_ids) < desired_instances:
                new_instances_needed = desired_instances - len(running_instance_ids)
                print(f"需要启动 {new_instances_needed} 个新实例")
                new_instance_ids = start_instances(ec2, new_instances_needed, ami_id, instance_type)
                running_instance_ids.extend(new_instance_ids)

            # 如果实例过多，终止多余实例
            elif len(running_instance_ids) > desired_instances:
                excess_instances = len(running_instance_ids) - desired_instances
                print(f"需要终止 {excess_instances} 个多余实例")
                terminate_instance_ids = running_instance_ids[:excess_instances]
                terminate_instances(ec2, terminate_instance_ids)
                running_instance_ids = running_instance_ids[excess_instances:]

        except Exception as e:
            print(f"调度过程中出现错误: {e}")

        # 等待一段时间后再次检查
        time.sleep(60)  # 每分钟检查一次

# 启动弹性调度器
if __name__ == "__main__":
    elastic_compute_scheduler()