#!/bin/bash
echo "This tool will help you convert .pt files to .bin files\n"
if [! -d "../docker"];then
	mkdir ../docker
	mkdir ../docker/my_workspace
fi
if [! -d "../samples"];then
	mkdir ../samples
fi
if [! -f "../docker/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz"];then
	echo "missing essential docker files...\n"
	if wget -P ../docker https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz; then
		echo "download success...."
	else
		echo "[ERROR]download failed!"
		exit 1
	fi
fi
if [! -f "../docker/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz"];then
	echo "missing essential OE tool-chain..."
	if wget -P ../docker https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz; then
		echo "download success...."
	else
		echo "[ERROR]download failed!"
		exit 1
	fi
fi
if [! -f "../samples/rdk_model_zoo"];then
	echo "missing MODEL ZOO..."
	if git clone https://github.com/D-Robotics/rdk_model_zoo.git ../samples
		echo "clone success...."
		git checkout rdk_x5
	else
		echo "[ERROR]clone failed!"
		exit 1
	fi
if [[ "$1" == *.pt ]];then
	echo ".pt file loaded"
else
	echo "[ERROR]it's not a .pt file!"
fi
python3 ../samples/rdk_model_zoo/samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py --pt $1
dir=$(dirname "$1")
basename=$(basename "$1" .pt)
onnx_file="$dir/$basename.onnx"
if [! -f "$onnx_file"];then
	echo "[ERROR].onnx file not found!"
	exit 1
else
	cp $onnxfile ../docker/my_workspace
fi
cp config.yaml ../docker/my_workspace
if docker run -it --rm -v $(pwd)/../docker/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz:/open_explorer -v $(pwd)/my_workspace:/workspace openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 /bin/bash -c "hb_mapper makertbin --config ../workspace/my_workspace/config.yaml --model-type onnxS"; then
	echo "docker run successful"
else
	echo "[ERROR]docker run failed, code: $?"
	exit 1
fi


