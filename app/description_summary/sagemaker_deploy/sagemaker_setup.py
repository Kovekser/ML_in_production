from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from app.config import config
import boto3
import logging
import sagemaker
import json
from time import sleep

logger = logging.getLogger(__name__)


class HuggingFaceModelDeploy:
    def __init__(self, role_name: str = None):
        boto3.set_stream_logger(name="botocore", level=logging.DEBUG)
        self.session = sagemaker.session.Session()
        if role_name:
            self.role = boto3.client('iam').get_role(RoleName=role_name)['Role']['Arn']
        else:
            self.role = sagemaker.get_execution_role(sagemaker_session=self.session)
        print(f"Role ARN: {self.role}")
        self.sm_client = boto3.client(service_name='sagemaker')
        self.sm_rt_client = boto3.client(service_name='sagemaker-runtime')
        self.model_name = sagemaker.utils.name_from_base(config.HUGGINGFACE__BASE_MODEL_ID.split("/")[-1].replace(".", ""))
        self.model_url = f"{config.SAGEMAKER.BUCKET}{config.HUGGINGFACE.BASE_MODEL_ID}"
        self.base_inference_component_name = f"base-{self.model_name}"
        self.endpoint_name = f"{self.model_name}-endpoint"

    def get_container_uri(self):
        try:
            container_uri = get_huggingface_llm_image_uri(
                backend="huggingface",
                region=config.SAGEMAKER__REGION,
            )
            logger.info(f"Retrieved container URI: {container_uri}")
            return container_uri
        except Exception as e:
            logger.error(f"Failed to get container URI: {str(e)}")
            raise



    def create_model_djl_lmi(self):
        try:
            image_uri = "763104351884.dkr.ecr.eu-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126"
            environment = {"HF_MODEL_ID": config.HUGGINGFACE.BASE_MODEL_ID,
                           'HUGGING_FACE_HUB_TOKEN': config.HUGGINGFACE.TOKEN,
                           "OPTION_ENABLE_LORA": "true",
                           "OPTION_ROLLING_BATCH": "disable",
                           "OPTION_TENSOR_PARALLEL_DEGREE": "1",
                           "OPTION_MAX_LORAS": "5",
                           "OPTION_DTYPE": "fp16",
                           "OPTION_MAX_LORA_RANK": "64",
                           "OPTION_MAX_CPU_LORAS": "2",
                           "OPTION_LOADFORMAT": "safetensors"}
            create_model_response = self.sm_client.create_model(
                ModelName=self.model_name,
                ExecutionRoleArn=self.role,
                PrimaryContainer={
                    "Image": image_uri,
                    "Environment": environment,
                },
            )
            logger.info(f"Successfully created model: {self.model_name}")
            return create_model_response
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise

    def create_model_tgi(self):
        try:
            container_uri = self.get_container_uri()

            # Environment variables for LORA configuration
            environment = {
                'HUGGING_FACE_HUB_TOKEN': config.HUGGINGFACE__TOKEN,
                'HF_TASK': "text-generation",
                'SM_NUM_GPUS': '1',
                'MAX_TOTAL_TOKENS': json.dumps(2048),
                'HF_MODEL_ID': config.HUGGINGFACE__BASE_MODEL_ID,
                'LORA_ADAPTERS': config.SAGEMAKER__ADAPTERS,
            }

            # Create model
            response = self.sm_client.create_model(
                ModelName=self.model_name,
                ExecutionRoleArn=self.role,
                Containers=[{
                    'Image': container_uri,
                    'Environment': environment,
                }]
            )
            logger.info(f"Successfully created model: {self.model_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise

    def create_endpoint_config(self):
        try:
            response = self.sm_client.create_endpoint_config(
                EndpointConfigName=f"{self.model_name}-config",
                ExecutionRoleArn=self.role,
                ProductionVariants=[{
                    'InstanceType': config.SAGEMAKER.INSTANCE_TYPE,
                    'InitialInstanceCount': 1,
                    # 'ModelName': self.model_name,
                    'VariantName': 'AllTraffic',
                    'ContainerStartupHealthCheckTimeoutInSeconds': 600,
                    'ModelDataDownloadTimeoutInSeconds': 900,
                    "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
                }]
            )
            logger.info(f"Successfully created endpoint config: {self.model_name}-config")
            return response
        except Exception as e:
            logger.error(f"Failed to create endpoint config: {str(e)}")
            raise

    def create_endpoint(self):
        """Create and deploy the endpoint"""
        try:
            response = self.sm_client.create_endpoint(
                EndpointName=self.endpoint_name,
                EndpointConfigName=f"{self.model_name}-config"
            )

            logger.info(f"Creating endpoint: {self.model_name}")

            # Wait for endpoint to be ready
            waiter = self.sm_client.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=self.endpoint_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )

            logger.info(f"Successfully created endpoint: {self.endpoint_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to create endpoint: {str(e)}")
            raise

    def _inference_component_completed(self, inference_component_name):
        while True:
            response = self.sm_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
            status = response["InferenceComponentStatus"]
            if status == 'InService':
                return True
            elif status in ["Failed", "Deleting"]:
                raise Exception(f"Inference Component failed or is being deleted: {status}")
            else:
                logger.info(f"Inference Component status: {status}")
                sleep(60)

    def create_base_inference_component(self):
        try:
            variant_name = "AllTraffic"

            inference_response = self.sm_client.create_inference_component(
                InferenceComponentName=self.base_inference_component_name,
                EndpointName=self.endpoint_name,
                VariantName=variant_name,
                Specification={
                    "ModelName": self.model_name,
                    "StartupParameters": {
                        "ModelDataDownloadTimeoutInSeconds": 900,
                        "ContainerStartupHealthCheckTimeoutInSeconds": 900,
                    },
                    "ComputeResourceRequirements": {
                        "MinMemoryRequiredInMb": 28000,
                        "NumberOfAcceleratorDevicesRequired": 1,
                    },
                },
                RuntimeConfig={
                    "CopyCount": 1,
                },
            )
            if self._inference_component_completed(self.base_inference_component_name):
                logger.info(f"Successfully created base inference component: {self.base_inference_component_name}")
                return inference_response
        except Exception as e:
            logger.error(f"Failed to create base inference component: {str(e)}")
            raise

    def create_summary_adapter(self):
        try:
            ic_name = "adapter-summarization-Llama-31-8B-Instruct"
            adapter_path = f"{config.SAGEMAKER.BUCKET}{config.SAGEMAKER.SUMMARY_ADAPTER}"
            logging.info(f"Creating summary adapter: {ic_name} from {adapter_path}")

            create_ic_response = self.sm_client.create_inference_component(
                InferenceComponentName=ic_name,
                EndpointName=self.endpoint_name,
                Specification={
                    "BaseInferenceComponentName": self.base_inference_component_name,
                    "Container": {
                        "ArtifactUrl": adapter_path
                    },
                },
            )
            if self._inference_component_completed(ic_name):
                logger.info(f"Successfully created summary adapter: {ic_name}")
                return create_ic_response
        except Exception as e:
            print(create_ic_response)
            logger.error(f"Failed to create summary adapter: {str(e)}")
            raise

    def deploy(self):
        """Deploy the complete multilingual support system"""
        try:
            logger.info("Starting deployment process...")
            self.create_model_djl_lmi()
            self.create_endpoint_config()
            self.create_endpoint()
            self.create_base_inference_component()
            self.create_summary_adapter()
            logger.info("Deployment completed successfully!")
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            self.cleanup()
            raise

    def cleanup(self):
        self.delete_inference_component()
        self.delete_endpoint()
        self.delete_endpoint_config()
        self.delete_model()


    def delete_endpoint(self):
        try:
            response = self.sm_client.delete_endpoint(
                EndpointName=self.endpoint_name
            )
            logger.info(f"Successfully deleted endpoint: {self.endpoint_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {str(e)}")
            pass

    def delete_endpoint_config(self):
        try:
            response = self.sm_client.delete_endpoint_config(
                EndpointConfigName=f"{self.model_name}-config"
            )
            logger.info(f"Successfully deleted endpoint config: {self.model_name}-config")
            return response
        except Exception as e:
            logger.error(f"Failed to delete endpoint config: {str(e)}")
            pass

    def delete_model(self):
        try:
            response = self.sm_client.delete_model(
                ModelName=self.model_name
            )
            logger.info(f"Successfully deleted model: {self.model_name}")
            return response
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            pass

    def delete_inference_component(self):
        try:
            for endpoint in self.sm_client.list_inference_components(
                    EndpointName=self.endpoint_name,
                    MaxResults=100,
            )["InferenceComponents"]:
                if endpoint["InferenceComponentStatus"] in ("InService", "Failed", "Deleting"):
                    logger.info(f"Deleting inference component: {endpoint['InferenceComponentName']}")
                    self.sm_client.delete_inference_component(
                        InferenceComponentName=endpoint["InferenceComponentName"],
                    )
                logger.info(f"Successfully deleted inference component: {self.base_inference_component_name}")
        except Exception as e:
            logger.error(f"Failed to delete inference component: {str(e)}")
            pass


def deploy_model():
    deploy = HuggingFaceModelDeploy(config.SAGEMAKER__ROLE_NAME)
    deploy.deploy()
