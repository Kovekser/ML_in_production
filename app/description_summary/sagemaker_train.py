import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
from app.config import config

env = {"HF_TOKEN": config.HUGGINGFACE.TOKEN,
       "WANDB_API_KEY": config.WANDB_API_KEY}

my_bucket_name = "ss-bucket-kovalenko-test"
prefix = "models/Llama-3.1-8B-Instruct/adapters/description_summaries"
s3_path = f"s3://{my_bucket_name}/{prefix}"

dataset_prefix = "datasets/descriptions_summaries"
train_path = f"s3://ss-bucket-kovalenko-test/{dataset_prefix}/train.json"
test_path = f"s3://ss-bucket-kovalenko-test/{dataset_prefix}/test.json"

config_path = "conf/gpu_train.json"

sess = sagemaker.Session(default_bucket=my_bucket_name)

iam = boto3.client('iam')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-test')['Role']['Arn']

print(f"sugemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

huggingface_estimator = HuggingFace(
    py_version="py311",
    base_job_name='huggingface-llama-8b',
    entry_point="train.py",
    source_dir="./app/description_summary",
    instance_type="ml.g5.xlarge",
    instance_count=1,
    transformers_version="4.46",
    role=role,
    pytorch_version="2.3",
    dependencies=["./app/description_summary/requirements.txt"],
    output_path=s3_path,
    environment=env,
    hyperparameters={
        "config_path": config_path  # Pass config file path
    },
)

huggingface_estimator.fit()


