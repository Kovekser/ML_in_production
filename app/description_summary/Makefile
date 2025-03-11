build:
	docker buildx build --platform linux/arm64 -f Dockerfile -t description-summary:latest .

run_dev: build
	docker run -it -v ${PWD}:/main description-summary:latest /bin/bash

run_dev_gpu: build
	docker run --net=host --gpus all -it -v ${PWD}:/main description-summary:latest /bin/bash