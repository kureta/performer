build:
	docker buildx build --tag kureta/performer:latest --target base -f Dockerfile .

run:
	docker run --mount type=bind,source="$(shell pwd)",target=/home/developer/app --name performer --gpus all --ipc=host -it --rm kureta/performer:latest bash
