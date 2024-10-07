# ML_in_production
Repository for education on Machine Learning in Production course

## Running minio locally (MacOS):
Install minio using brew:
```shell
brew install minio
```
Run minio server. Run in command line:
```shell
/opt/homebrew/opt/minio/bin/minio server --certs-dir\=/opt/homebrew/etc/minio/certs --address\=:9000 /opt/homebrew/var/minio
```
To access the minio dashboard, open the browser and go to http://localhost:9000. The default credentials are:
```shell
login: minioadmin
password: minioadmin
```

## Running minio in Docker:
To rnu minio in docker, run the following command:
```shell
mkdir -p ${HOME}/minio/data

docker run \
   -p 9000:9000 \
   -p 9001:9001 \
   --user $(id -u):$(id -g) \
   --name minio1 \
   -e "MINIO_ROOT_USER=ROOTUSER" \
   -e "MINIO_ROOT_PASSWORD=CHANGEME123" \
   -v ${HOME}/minio/data:/data \
   quay.io/minio/minio server /data --console-address ":9001"
```
Access the MinIO Console by going to a browser and going to http://127.0.0.1:9000. The default credentials are:
```shell
login: ROOTUSER
password: CHANGEME123
```

## Running minio in k8s cluster:
In terminal run kubectl commands:
```shell
kubectl apply -f minio.pvc.yaml
kubectl apply -f minio-deployment.yaml
kubectl apply -f minio-service.yaml
``` 
This will deploy minio in the k8s cluster. To access the minio dashboard, run the following command:
```shell
kubectl port-forward service/minio-service 7000:9000
kubectl port-forward service/minio-service 42933:42933
```
Then open the browser and go to http://localhost:7000. The default credentials are:
```shell
login: minio
password: minio123
```
After login object storage is ready to be used in the k8s cluster.