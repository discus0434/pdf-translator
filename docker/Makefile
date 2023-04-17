NAME=pdf-translator
TAG=0.1.0
PROJECT_DIRECTORY=$(shell pwd)/..

build:
	docker build -t ${NAME}:${TAG} .

run:
	docker run -it \
		--runtime=nvidia \
		--name pdf-translator \
		-v ${PROJECT_DIRECTORY}:/home/${NAME} \
		--gpus all \
		-d --restart=always \
		-p 8765:8765 \
		${NAME}:${TAG} /bin/bash

translate:
	@cd ${PROJECT_DIRECTORY} && python3 translator.py -i ${INPUT}
