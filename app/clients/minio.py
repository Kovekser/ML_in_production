from minio import Minio, deleteobjects
from app.config import config
from typing import Optional, List, Any, Dict
from utils import Progress


class MinioClient:
    def __init__(self):
        self.minio_url = config.MINIO.ENDPOINT
        self.access_key = config.MINIO.ACCESS_KEY
        self.secret_key = config.MINIO.SECRET_KEY
        self.client = Minio(self.minio_url, self.access_key, self.secret_key)

    def get_all_objects(
        self, bucket_name: str, prefix: Optional[str] = None, recursive: Optional[bool] = False
    ) -> List[Any]:
        return self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)

    def delete_one_object(self, bucket_name: str, object_name: str) -> None:
        self.client.remove_object(bucket_name, object_name)

    def delete_many_objects(self, bucket_name: str, prefix: Optional[str] = None, recursive: Optional[bool] = False) -> None:
        delete_obj_list = map(
            lambda x: deleteobjects.DeleteObject(x.object_name),
            self.get_all_objects(bucket_name, prefix=prefix, recursive=recursive),
        )
        self.client.remove_objects(bucket_name, delete_obj_list)

    def download_file(self, bucket_name: str, object_name: str, file_path: str, version: Optional[str] = None) -> Any:
        return self.client.fget_object(bucket_name, object_name, file_path, version_id=version, progress=Progress)

    def upload_file(
        self, bucket_name: str, file_path: str, object_name: str, data: Any, length: int, metadata: Dict[str, Any] = None
    ) -> Any:
        return self.client.fput_object(bucket_name, object_name, file_path, metadata=metadata, progress=Progress)
