NAME=pdf-translator
TAG=0.1.0
CPU_TAG=0.1.0-cpu
CPU_HIFI_TAG=0.1.0-cpu-hifi
PROJECT_DIRECTORY=$(shell pwd)/..
CONTAINER_NAME=pdf-translator
CPU_CONTAINER_NAME=pdf-translator-cpu
CPU_HIFI_CONTAINER_NAME=pdf-translator-cpu-hifi

build:
	docker build -t ${NAME}:${TAG} .

build-cpu:
	docker build -f Dockerfile.cpu -t ${NAME}:${CPU_TAG} .

build-cpu-hifi:
	docker build -f Dockerfile.cpu-hifi -t ${NAME}:${CPU_HIFI_TAG} .

run:
	docker run -it \
		--runtime=nvidia \
		--name ${CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		--gpus all \
		-d --restart=always \
		-p 8765:8765 \
		-p 8288:8288 \
		${NAME}:${TAG} /bin/bash -c "chmod +x /app/docker/start.sh && /app/docker/start.sh"

run-cpu:
	docker run -it \
		--name ${CPU_CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		-d --restart=always \
		-p 8765:8765 \
		-p 8288:8288 \
		-e PDF_TRANSLATOR_DEVICE=auto \
		-e PDF_TRANSLATOR_LAYOUT_BACKEND=none \
		${NAME}:${CPU_TAG} /bin/bash -c "chmod +x /app/docker/start-cpu.sh && /app/docker/start-cpu.sh"

run-cpu-hifi:
	docker run -it \
		--name ${CPU_HIFI_CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		-d --restart=always \
		-p 9765:8765 \
		-p 9288:8288 \
		-e PDF_TRANSLATOR_DEVICE=cpu \
		-e PDF_TRANSLATOR_LAYOUT_BACKEND=detectron2 \
		${NAME}:${CPU_HIFI_TAG} /bin/bash -c "chmod +x /app/docker/start-cpu-hifi.sh && /app/docker/start-cpu-hifi.sh"

run-bash:
	docker run -it \
		--runtime=nvidia \
		--name ${CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		--gpus all \
		-d --restart=always \
		-p 8765:8765 \
		-p 8288:8288 \
		${NAME}:${TAG} /bin/bash

run-bash-cpu:
	docker run -it \
		--name ${CPU_CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		-d --restart=always \
		-p 8765:8765 \
		-p 8288:8288 \
		${NAME}:${CPU_TAG} /bin/bash

run-bash-cpu-hifi:
	docker run -it \
		--name ${CPU_HIFI_CONTAINER_NAME} \
		-v ${PROJECT_DIRECTORY}:/app \
		-d --restart=always \
		-p 9765:8765 \
		-p 9288:8288 \
		${NAME}:${CPU_HIFI_TAG} /bin/bash

local-run:
	@cd ${PROJECT_DIRECTORY} && \
		export PDF_TRANSLATOR_MODEL_ROOT_DIR=${PROJECT_DIRECTORY}/models && \
		export PDF_TRANSLATOR_DEVICE=auto && \
		export PDF_TRANSLATOR_LAYOUT_BACKEND=none && \
		python3 gui.py & \
		cd server && \
		python3 main.py

local-run-api:
	@cd ${PROJECT_DIRECTORY} && \
		export PDF_TRANSLATOR_MODEL_ROOT_DIR=${PROJECT_DIRECTORY}/models && \
		export PDF_TRANSLATOR_DEVICE=auto && \
		export PDF_TRANSLATOR_LAYOUT_BACKEND=none && \
		python3 server/main.py

local-run-gui:
	@cd ${PROJECT_DIRECTORY} && \
		python3 gui.py

logs:
	docker logs -f ${CONTAINER_NAME}

logs-cpu:
	docker logs -f ${CPU_CONTAINER_NAME}

logs-cpu-hifi:
	docker logs -f ${CPU_HIFI_CONTAINER_NAME}

stop:
	-docker stop ${CONTAINER_NAME}

stop-cpu:
	-docker stop ${CPU_CONTAINER_NAME}

stop-cpu-hifi:
	-docker stop ${CPU_HIFI_CONTAINER_NAME}

rm:
	-docker rm ${CONTAINER_NAME}

rm-cpu:
	-docker rm ${CPU_CONTAINER_NAME}

rm-cpu-hifi:
	-docker rm ${CPU_HIFI_CONTAINER_NAME}

clean:
	-docker stop ${CONTAINER_NAME}
	-docker rm ${CONTAINER_NAME}

clean-cpu:
	-docker stop ${CPU_CONTAINER_NAME}
	-docker rm ${CPU_CONTAINER_NAME}

clean-cpu-hifi:
	-docker stop ${CPU_HIFI_CONTAINER_NAME}
	-docker rm ${CPU_HIFI_CONTAINER_NAME}

translate:
	@cd ${PROJECT_DIRECTORY} && \
		python3 cli.py -i ${INPUT}
