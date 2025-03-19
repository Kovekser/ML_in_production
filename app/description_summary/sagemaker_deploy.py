from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from app.config import config
import boto3

iam = boto3.client('iam')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-test')['Role']['Arn']
print(f"SageMaker execution role arn: {role}")

env = {
    'HF_TASK': "text-generation",
    'HUGGING_FACE_HUB_TOKEN': config.HF_TOKEN,
    "MODEL_CONFIG_FILE": "config.json",
}
image_uri = get_huggingface_llm_image_uri(
    backend="huggingface",
    region="eu-west-2",
)
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    env=env,
    model_data=config.PATH_TO_MODEL,  # Change to your model path
    role=role,
    model_server_workers=1,
    image_uri=image_uri,
)

predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type= "ml.g5.2xlarge"
)
