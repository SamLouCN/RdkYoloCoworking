## This is a YOLO-based object detection project deployed on the RDK x5 platform, specifically targeting sea stars and sea cucumbers.
Training Device Info: <br>
CPU: AMD Ryzen R7 9700X<br>
GPU: AMD Radeon RX 9070XT<br>

## How to use:
#### Step1: Training
Put your dataset in the root directory<br>
Configure your `data.yaml`<br>
Run `training.py`, your result pt file will be saved at runs/detect
```bash
python3 training.py
```
#### Step2: Converting
Check `config.yaml`, and customize your binary file output configurations<br>
Run convert.sh, your result bin file will be saved at docker/my_workspace as default
```bash
cd shell
./convert.sh /dir/to/pt/file
```
==Warning==<br>
Avoid arbitrary modifications to the config.yaml file. Since the conversion process runs inside a Docker container, the internal directory structure may not correspond directly to the host's file system, so paths and mappings can be inconsistent.
#### Step3: Deploying
Clone this repository to a RDK x5, and run doRdkYolo.py, edit doRdkYolo.py and set
```python
model_path="/dir/to/bin/file"
```
Then, run this py file
```bash
python3 doRdkYolo.py
```