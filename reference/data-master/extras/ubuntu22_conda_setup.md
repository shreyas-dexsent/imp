# Ubuntu 22.04 Conda Setup

This workspace contains these local repos:

- `~/imp/dexsent-vgr-camera-core-2.5d`
- `~/imp/dexsent-vgr-vision-engine-2.5d`
- `~/imp/dexsent-vgr-robot-engine`
- `~/imp/dexsent-vgr-application-orchestrator`

Note:

- Your quick-run snippet uses `~/vgr/...` paths. In this workspace, the repos are under `~/imp/...`.
- `configs/robot.local.json` keeps FR3 as the default robot.
- `configs/robot.xarm.json` is available as a backup xArm Lite6 config and currently points to `192.168.1.111`; change that host if your xArm uses a different IP.

## 0. Base Ubuntu Packages

```bash
sudo apt update
sudo apt install -y build-essential pkg-config git curl wget \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
  libegl1 libopengl0 libglfw3 libglew2.2
```

If `conda` is not installed yet, install Miniconda first, then restart your shell.

## 1. Camera Env: `vgr-camera`

Repo: `~/imp/dexsent-vgr-camera-core-2.5d`

```bash
conda create -n vgr-camera python=3.9 -y
conda activate vgr-camera
cd ~/imp/dexsent-vgr-camera-core-2.5d
pip install -r requirements.txt
```

Run:

```bash
conda activate vgr-camera
cd ~/imp/dexsent-vgr-camera-core-2.5d
python -m camera_core.main --config config/cam_realsense_d435i.yaml
```

Optional:

```bash
conda activate vgr-camera
cd ~/imp/dexsent-vgr-camera-core-2.5d
python -m camera_core.main --config config/cam_webcam.yaml
```

## 2. Vision Env: `vgr-vision`

Repo: `~/imp/dexsent-vgr-vision-engine-2.5d`

```bash
conda create -n vgr-vision python=3.11 -y
conda activate vgr-vision
cd ~/imp/dexsent-vgr-vision-engine-2.5d
pip install -r requirements.txt
```

If you already installed the vision env before this compatibility pin, repair it once with:

```bash
conda activate vgr-vision
cd ~/imp/dexsent-vgr-vision-engine-2.5d
pip install --upgrade --force-reinstall "numpy>=1.26,<2" "opencv-python>=4.10,<4.11"
pip install -r requirements.txt
```

Run:

```bash
conda activate vgr-vision
cd ~/imp/dexsent-vgr-vision-engine-2.5d
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Note:

- This repo installs `torch`, `torchvision`, `open3d`, `panda3d`, and `ultralytics`, so it is the heaviest env.
- If you need a CUDA-specific PyTorch build, install the matching `torch` and `torchvision` first, then run `pip install -r requirements.txt`.
- `megapose_bin_picking` currently depends on NumPy 1.x-era MegaPose code, so the vision env is intentionally pinned to `numpy<2` and `opencv-python<4.11` to avoid the NumPy 2 / OpenCV 4.12 conflict.

## 3. Robot Env: `vgr-robot`

Repo: `~/imp/dexsent-vgr-robot-engine`

```bash
conda create -n vgr-robot python=3.11 -y
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
```

Note:

- The FR3 adapter now uses the `franky` Python module from `franky-control`.
- The same env also includes the official `xarm-python-sdk`, so `configs/robot.xarm.json` can be used as a backup backend without creating a second robot env.
- For your current FR3 server version `9`, install `libfranka 0.15.3` first, then build `franky-control` against it.
- If you previously installed `pylibfranka` or a wheel-built `franky-control`, remove them before installing the server-9 stack:
```bash
conda activate vgr-robot
pip uninstall -y pylibfranka franky-control
```

Install the system-side Franka dependencies and `libfranka 0.15.3`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl lsb-release \
  libpoco-dev libeigen3-dev libfmt-dev
sudo mkdir -p /etc/apt/keyrings
curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc >/dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt-get update
sudo apt-get install -y robotpkg-pinocchio
echo "/opt/openrobots/lib" | sudo tee /etc/ld.so.conf.d/robotpkg-openrobots.conf >/dev/null
sudo ldconfig
sudo apt-get remove -y "*libfranka*" || true
cd /tmp
rm -rf libfranka
git clone --recurse-submodules https://github.com/frankarobotics/libfranka.git
cd libfranka
git checkout 0.15.3
git submodule update --init --recursive
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake -DBUILD_TESTS=OFF ..
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

Then install the robot repo requirements, which build `franky-control` against that local `libfranka`:

```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
pip install "pybind11>=2.11,<3"
export pybind11_DIR="$(python -m pybind11 --cmakedir)"
export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH}"
CMAKE_PREFIX_PATH="/usr/local;/opt/openrobots/lib/cmake;${pybind11_DIR}" \
  pip install --no-build-isolation -r requirements.txt
```

Why `0.15.3`:

- Franka's official `libfranka` tags `0.15.2` and `0.15.3` both use the protocol-9 `libfranka-common` line.
- `franky` upstream documents testing against `libfranka 0.15.3`, so that is the safer target for a server-version-9 robot.
- Current `pylibfranka` wheels are on a newer protocol and will not connect to your robot.
- Ubuntu 22's `pybind11-dev` package is `2.9.1`, which is too old for this Python-3.11 build path, so use the pip-installed `pybind11` and `--no-build-isolation`.
- The `franky` stub-generation step imports the built module during install, so `/opt/openrobots/lib` must be on the runtime linker path for Pinocchio.

Prepare the UR5e MuJoCo model:

```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
python tools/setup_ur5e_mjcf.py
```

Run:

```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
python app.py --config configs/robot.local.json
```

Note:

- `configs/robot.local.json` is present in this repo.
- `configs/robot.xarm.json` is present as a backup xArm Lite6 config.
- Install the xArm SDK into the same env with:
```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
pip install xarm-python-sdk
```
- Start the backup xArm path with:
```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
python app.py --config configs/robot.xarm.json
```

## 4. Orchestrator Env: `vgr-orch`

Repo: `~/imp/dexsent-vgr-application-orchestrator`

```bash
conda create -n vgr-orch python=3.11 -y
conda activate vgr-orch
cd ~/imp/dexsent-vgr-application-orchestrator
pip install -r requirements.txt
```

Run:

```bash
conda activate vgr-orch
cd ~/imp/dexsent-vgr-application-orchestrator
python -m uvicorn app:app --host 127.0.0.1 --port 8100
```

## 5. One-Time Setup Commands Together

Run these once, one block at a time:

```bash
conda create -n vgr-camera python=3.9 -y
conda activate vgr-camera
cd ~/imp/dexsent-vgr-camera-core-2.5d
pip install -r requirements.txt
```

```bash
conda create -n vgr-vision python=3.11 -y
conda activate vgr-vision
cd ~/imp/dexsent-vgr-vision-engine-2.5d
pip install -r requirements.txt
```

```bash
conda create -n vgr-robot python=3.11 -y
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
pip install "pybind11>=2.11,<3"
export pybind11_DIR="$(python -m pybind11 --cmakedir)"
export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH}"
CMAKE_PREFIX_PATH="/usr/local;/opt/openrobots/lib/cmake;${pybind11_DIR}" \
  pip install --no-build-isolation -r requirements.txt
python tools/setup_ur5e_mjcf.py
```

```bash
conda create -n vgr-orch python=3.11 -y
conda activate vgr-orch
cd ~/imp/dexsent-vgr-application-orchestrator
pip install -r requirements.txt
```

## 6. Start All 4 Services

Open 4 terminals and run:

```bash
conda activate vgr-camera
cd ~/imp/dexsent-vgr-camera-core-2.5d
python -m camera_core.main --config config/cam_realsense_d435i.yaml
```

```bash
conda activate vgr-vision
cd ~/imp/dexsent-vgr-vision-engine-2.5d
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

```bash
conda activate vgr-robot
cd ~/imp/dexsent-vgr-robot-engine
python app.py --config configs/robot.local.json
```

```bash
conda activate vgr-orch
cd ~/imp/dexsent-vgr-application-orchestrator
python -m uvicorn app:app --host 127.0.0.1 --port 8100
```
