from minio import Minio, deleteobjects
from app.config import config
from typing import Optional, List, Any, Dict
from utils import Progress


class MinioClient:
    def __init__(self, http_client=None):
        self.minio_url = f"{config.MINIO.HOST}:{config.MINIO.PORT}"
        self.access_key = config.MINIO.ACCESS_KEY
        self.secret_key = config.MINIO.SECRET_KEY
        self.http_client = http_client
        self.client = Minio(
            endpoint=self.minio_url,
            access_key=self.access_key,
            secret_key=self.secret_key,
            http_client=self.http_client,
)

    def get_all_objects(
        self, bucket_name: str, prefix: Optional[str] = None, recursive: Optional[bool] = False
    ) -> List[Any]:
        self.ensure_bucket_exists(bucket_name)
        return self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)

    def delete_one_object(self, bucket_name: str, object_name: str) -> None:
        self.ensure_bucket_exists(bucket_name)
        self.client.remove_object(bucket_name, object_name)

    def delete_many_objects(self, bucket_name: str, prefix: Optional[str] = None, recursive: Optional[bool] = False) -> None:
        self.ensure_bucket_exists(bucket_name)
        delete_obj_list = map(
            lambda x: deleteobjects.DeleteObject(x.object_name),
            self.get_all_objects(bucket_name, prefix=prefix, recursive=recursive),
        )
        self.client.remove_objects(bucket_name, delete_obj_list)

    def download_file(self, bucket_name: str, object_name: str, file_path: str, version: Optional[str] = None) -> Any:
        self.ensure_bucket_exists(bucket_name)
        return self.client.fget_object(bucket_name, object_name, file_path, version_id=version, progress=Progress)

    def upload_file(
        self, bucket_name: str, file_path: str, object_name: str, metadata: Dict[str, Any] = None
    ) -> Any:
        self.ensure_bucket_exists(bucket_name)
        return self.client.fput_object(bucket_name, object_name, file_path, metadata=metadata, progress=Progress)

    def ensure_bucket_exists(self, bucket_name: str) -> None:
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)


def get_minio_client():
    return MinioClient()
