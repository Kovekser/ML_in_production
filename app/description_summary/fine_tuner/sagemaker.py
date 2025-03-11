from sagemaker.s3 import S3Uploader
import sagemaker
import random



# If a default bucket prefix is specified, append it to the s3 path


local_data_file = "train.jsonl"
S3Uploader.upload(local_data_file, s3_location)
S3Uploader.upload("template.json", train_data_location)
print(f"Training data: {train_data_location}")

def upload_dataset_to_s3(local_data_file_path: str, s3_name: str):

    output_bucket = sagemaker.Session().default_bucket()
    default_bucket_prefix = sagemaker.Session().default_bucket_prefix

    if default_bucket_prefix:
        train_data_location = f"s3://{output_bucket}/{default_bucket_prefix}/dolly_dataset"
    else:
        train_data_location = f"s3://{output_bucket}/dolly_dataset"
