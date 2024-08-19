def test_delete_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "test.txt")
    minio_client.delete_one_object("testbucket", "test.txt")
    result = minio_client.get_all_objects("testbucket")
    assert len(result) == 0


def test_get_list_of_files(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "test.txt")
    result = minio_client.get_all_objects("testbucket")
    assert len(result) == 1
    assert result[0].object_name == "test.txt"


def test_delete_many_files(minio_client):
    minio_client.upload_file("testbucket", "test1.txt", "test.txt")
    minio_client.upload_file("testbucket", "test2.txt", "test.txt")
    minio_client.upload_file("testbucket", "test3.txt", "test.txt")
    minio_client.delete_many_objects("testbucket")
    result = minio_client.get_all_objects("testbucket")
    assert len(result) == 0


def test_upload_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "test.txt")
    result = minio_client.get_all_objects("testbucket")
    assert len(result) == 1
    assert result[0].object_name == "test.txt"


def test_download_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "test.txt")
    result = minio_client.download_file("testbucket", "test.txt", "test.txt")
    assert result is not None
