def test_delete_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "tests/test.txt")
    minio_client.delete_one_object("testbucket", "test.txt")
    result = list(minio_client.get_all_objects("testbucket"))
    assert len(result) == 0


def test_get_list_of_files(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "tests/test.txt")
    result = list(minio_client.get_all_objects("testbucket"))
    assert len(result) == 1
    assert result[0].object_name == "test.txt"


def test_delete_many_files(minio_client):
    minio_client.upload_file("testbucket", "test1.txt","tests/test.txt")
    minio_client.upload_file("testbucket", "test2.txt", "tests/test.txt")
    minio_client.upload_file("testbucket", "test3.txt", "tests/test.txt", )
    assert len(list(minio_client.get_all_objects("testbucket"))) == 3
    minio_client.delete_many_objects("testbucket")
    result = list(minio_client.get_all_objects("testbucket"))
    assert len(result) == 0


def test_upload_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "tests/test.txt")
    result = list(minio_client.get_all_objects("testbucket"))
    assert len(result) == 1
    assert result[0].object_name == "test.txt"


def test_download_file(minio_client):
    minio_client.upload_file("testbucket", "test.txt", "tests/test.txt")
    result = minio_client.download_file("testbucket", "test.txt", "tests/test.txt")
    assert result is not None
