# Forearm US
 pictures of forearm ultrasound for Robot Hand Teleoperation

## 2025-02-27
 - first time v10 trial
 - I have uploaded some test pictures
 - made codec converter code and image cropping code

## 2025-04-04
 - 6가지 동작 분류를 위한 실험을 진행함.

## 2025-07-09
 - 베라소닉스 사용해서 실시간 연동 진행함.
 - matlab 파일에 User 버튼 달아 시작/중지 토글 스위치 만들었음.
 - USB 환경에서 MuJoCo와 Python 다운로드 받아 동작해서 실험실 컴퓨터는 uncontaminated 상태 유지함.

좋아요! Jetson Nano에서 Docker로 ROS2 환경을 만드는 방법을 차근차근 알려드릴게요.

## 단계 1: 작업 폴더 만들기

터미널을 열고 다음을 입력하세요:

```bash
# 홈 디렉토리로 이동
cd ~

# ROS2 작업용 폴더 만들기
mkdir ros2_docker
cd ros2_docker

# 작업공간 폴더도 미리 만들어두기
mkdir workspace
```

## 단계 2: Dockerfile 만들기

파일을 만들어야 해요. 텍스트 에디터를 사용합니다:

```bash
# nano 에디터로 Dockerfile 만들기
nano Dockerfile
```

아래 내용을 **그대로** 복사해서 붙여넣으세요:

```dockerfile
FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# 기본 도구들 설치
RUN apt update && apt install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    python3-pip \
    wget

# ROS2 설치 준비
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS2 설치
RUN apt update && apt install -y \
    ros-humble-desktop \
    python3-argcomplete \
    python3-colcon-common-extensions

# 작업 폴더 설정
WORKDIR /workspace

# ROS2 자동 실행 설정
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

저장하고 나가기: `Ctrl + X` → `Y` → `Enter`

## 단계 3: Docker 이미지 만들기

```bash
# 이미지 빌드 (시간이 좀 걸려요, 10-20분 정도)
docker build -t my-ros2 .
```

빌드가 진행되는 동안 기다리세요. 에러가 나면 멈추고 알려주세요!

## 단계 4: Docker 컨테이너 실행하기

### 4-1. 간단한 실행 (터미널만)

```bash
docker run -it --rm my-ros2 bash
```

### 4-2. 작업 폴더와 연결해서 실행

```bash
docker run -it --rm \
    -v ~/ros2_docker/workspace:/workspace \
    my-ros2 bash
```

## 단계 5: ROS2 동작 확인

컨테이너 안에서 (프롬프트가 `root@...:/workspace#` 형태로 바뀜):

```bash
# ROS2가 제대로 설치되었는지 확인
ros2 --help

# 간단한 테스트
ros2 run demo_nodes_cpp talker
```

다른 메시지들이 계속 출력되면 성공! `Ctrl + C`로 중단하세요.

## 단계 6: 편리한 실행 스크립트 만들기

매번 긴 명령어 치기 귀찮으니까 스크립트를 만들어봐요:

```bash
# 원래 터미널로 돌아가기 (exit로 컨테이너에서 나오기)
exit

# 스크립트 파일 만들기
nano run_ros2.sh
```

다음 내용을 입력:

```bash
#!/bin/bash
docker run -it --rm \
    -v ~/ros2_docker/workspace:/workspace \
    --name my-ros2-container \
    my-ros2 bash
```

저장하고 실행 권한 주기:

```bash
chmod +x run_ros2.sh

# 이제 이렇게 실행하면 돼요
./run_ros2.sh
```

## 문제 해결

**Docker 권한 에러가 나면:**
```bash
sudo usermod -aG docker $USER
# 그리고 재부팅하거나 다시 로그인
```

**빌드가 실패하면:**
- 인터넷 연결 확인
- 용량 부족 확인: `df -h`

**컨테이너에서 나가려면:**
```bash
exit
```

이제 Docker 안에서 ROS2 Humble을 사용할 수 있어요! 다음에 뭘 해보고 싶으신가요?



### 01010

사용자가 더 필요할 것 같은 패키지들을 모두 포함한 완전한 Dockerfile을 요청했다. 로봇 개발에서 사용할 수 있는 거의 모든 패키지를 포함해서 정말 포괄적인 Docker 이미지를 만들어보자.

다음과 같은 카테고리들을 고려해야 한다:
1. ROS2 기본 및 개발 도구
2. ROS2 Control 시스템
3. 로봇 모델링 및 시각화
4. 시뮬레이션 (Gazebo, Ignition)
5. 네비게이션 및 SLAM
6. 센서 드라이버들 (카메라, 라이다, IMU 등)
7. MoveIt (매니퓰레이터 제어)
8. 통신 (네트워크, 시리얼, CAN 등)
9. 컴퓨터 비전 및 머신러닝
10. 유틸리티 및 디버깅 도구
11. C++ 라이브러리들
12. Python 라이브러리들
13. 하드웨어 인터페이스들

정말 포괄적으로 만들어보자.

네! 정말 포괄적으로 로봇 개발에 필요할 수 있는 거의 모든 패키지를 포함한 Dockerfile을 만들어드릴게요:

```dockerfile
FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ===== 시스템 기본 설정 =====
RUN apt update && apt install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    vim \
    nano \
    htop \
    tree \
    unzip \
    zip \
    gdb \
    valgrind \
    clang \
    clang-format \
    pkg-config

# ===== ROS2 저장소 설정 =====
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ===== ROS2 코어 및 개발 도구 =====
RUN apt update && apt install -y \
    ros-humble-desktop-full \
    ros-humble-dev-tools \
    ros-humble-ros-base \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-rosinstall-generator \
    python3-wstool

# ===== ROS2 Control 시스템 (완전한 세트) =====
RUN apt install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-hardware-interface \
    ros-humble-controller-manager \
    ros-humble-control-msgs \
    ros-humble-control-toolbox \
    ros-humble-realtime-tools \
    ros-humble-transmission-interface \
    ros-humble-joint-limits \
    ros-humble-joint-limits-interface \
    ros-humble-force-torque-sensor-broadcaster \
    ros-humble-imu-sensor-broadcaster \
    ros-humble-effort-controllers \
    ros-humble-velocity-controllers \
    ros-humble-position-controllers \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller

# ===== 로봇 모델링, URDF, 시각화 =====
RUN apt install -y \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-urdf \
    ros-humble-urdf-launch \
    ros-humble-urdf-parser-plugin \
    ros-humble-robot-localization \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf2-tools \
    ros-humble-tf2-eigen

# ===== 시뮬레이션 (Gazebo + Ignition) =====
RUN apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-gazebo-plugins \
    ros-humble-gazebo-msgs \
    ros-humble-ignition-gazebo \
    ros-humble-ros-ign-bridge \
    ros-humble-ros-ign-gazebo

# ===== 시각화 및 GUI 도구 =====
RUN apt install -y \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-rviz-visual-tools \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-graph \
    ros-humble-rqt-console \
    ros-humble-rqt-plot \
    ros-humble-rqt-image-view \
    ros-humble-rqt-tf-tree \
    ros-humble-plotjuggler-ros

# ===== 네비게이션 및 SLAM =====
RUN apt install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-map-server \
    ros-humble-nav2-amcl \
    ros-humble-nav2-planner \
    ros-humble-nav2-controller \
    ros-humble-nav2-recoveries \
    ros-humble-nav2-bt-navigator \
    ros-humble-nav2-costmap-2d \
    ros-humble-slam-toolbox \
    ros-humble-cartographer \
    ros-humble-cartographer-ros \
    ros-humble-gmapping \
    ros-humble-hector-slam

# ===== MoveIt (매니퓰레이터 제어) =====
RUN apt install -y \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-moveit-fake-controller-manager \
    ros-humble-moveit-servo \
    ros-humble-moveit-visual-tools

# ===== 센서 드라이버 및 메시지 =====
RUN apt install -y \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-visualization-msgs \
    ros-humble-diagnostic-msgs \
    ros-humble-diagnostic-updater \
    ros-humble-diagnostic-aggregator \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-theora-image-transport

# ===== 카메라 및 비전 센서 =====
RUN apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-geometry \
    ros-humble-camera-calibration-parsers \
    ros-humble-camera-info-manager \
    ros-humble-usb-cam \
    ros-humble-v4l2-camera \
    ros-humble-realsense2-camera \
    ros-humble-depth-image-proc \
    ros-humble-image-proc \
    ros-humble-image-pipeline \
    ros-humble-stereo-image-proc

# ===== 라이다 및 포인트클라우드 =====
RUN apt install -y \
    ros-humble-laser-geometry \
    ros-humble-laser-filters \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-pcl-msgs \
    ros-humble-velodyne \
    ros-humble-velodyne-driver \
    ros-humble-velodyne-msgs \
    ros-humble-sick-scan2 \
    ros-humble-urg-node

# ===== IMU 및 기타 센서 =====
RUN apt install -y \
    ros-humble-imu-tools \
    ros-humble-imu-filter-madgwick \
    ros-humble-imu-complementary-filter \
    ros-humble-microstrain-inertial-driver \
    ros-humble-phidgets-drivers \
    ros-humble-spatio-temporal-voxel-layer \
    ros-humble-range-sensor-layer

# ===== 통신 및 네트워킹 =====
RUN apt install -y \
    ros-humble-serial-driver \
    ros-humble-rosbridge-suite \
    ros-humble-rosbridge-server \
    ros-humble-mqtt-client \
    ros-humble-socketcan-interface \
    ros-humble-can-msgs \
    ros-humble-udp-msgs \
    ros-humble-async-web-server-cpp \
    ros-humble-web-video-server

# ===== 로깅 및 데이터 저장 =====
RUN apt install -y \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage \
    ros-humble-rosbag2-compression \
    ros-humble-rosbag2-transport \
    ros-humble-sqlite3-vendor \
    ros-humble-mcap-vendor

# ===== Launch 및 라이프사이클 =====
RUN apt install -y \
    ros-humble-launch \
    ros-humble-launch-ros \
    ros-humble-launch-xml \
    ros-humble-launch-yaml \
    ros-humble-launch-testing \
    ros-humble-lifecycle-msgs \
    ros-humble-rclcpp \
    ros-humble-rclcpp-lifecycle \
    ros-humble-rclpy \
    ros-humble-rclcpp-components \
    ros-humble-composition

# ===== 테스팅 및 품질 도구 =====
RUN apt install -y \
    ros-humble-test-msgs \
    ros-humble-example-interfaces \
    ros-humble-demo-nodes-cpp \
    ros-humble-demo-nodes-py \
    ros-humble-ament-lint \
    ros-humble-ament-cmake-gtest \
    ros-humble-ament-cmake-gmock

# ===== C++ 라이브러리들 =====
RUN apt install -y \
    libeigen3-dev \
    libboost-all-dev \
    libpcl-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    libgmock-dev \
    libjsoncpp-dev \
    libceres-dev \
    libg2o-dev \
    libsuitesparse-dev \
    libtbb-dev \
    libfmt-dev \
    libspdlog-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libasio-dev \
    libssl-dev \
    libcrypto++-dev

# ===== 네트워크 및 하드웨어 라이브러리 =====
RUN apt install -y \
    libserial-dev \
    libusb-1.0-0-dev \
    libftdi1-dev \
    can-utils \
    iproute2 \
    net-tools \
    wireless-tools \
    bluez \
    libbluetooth-dev \
    libi2c-dev \
    libgpiod-dev

# ===== Python 패키지들 =====
RUN pip3 install \
    pytest \
    pytest-cov \
    pytest-mock \
    flake8 \
    mypy \
    black \
    isort \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    opencv-python \
    pillow \
    transforms3d \
    pyserial \
    pyyaml \
    requests \
    flask \
    fastapi \
    uvicorn \
    websockets \
    paho-mqtt \
    can \
    pymodbus \
    keyboard \
    psutil \
    gpustat

# ===== 추가 유틸리티 =====
RUN apt install -y \
    screen \
    tmux \
    terminator \
    x11-apps \
    mesa-utils \
    firefox \
    gedit \
    code

# ===== rosdep 초기화 =====
RUN rosdep init && rosdep update

# ===== 작업 환경 설정 =====
WORKDIR /workspace

# ===== 환경 변수 및 별명 설정 =====
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc && \
    echo "alias ll='ls -la'" >> ~/.bashrc && \
    echo "alias la='ls -la'" >> ~/.bashrc && \
    echo "alias cb='colcon build'" >> ~/.bashrc && \
    echo "alias cbs='colcon build --symlink-install'" >> ~/.bashrc && \
    echo "alias ct='colcon test'" >> ~/.bashrc && \
    echo "alias cr='colcon build && source install/setup.bash'" >> ~/.bashrc && \
    echo "alias rr='ros2 run'" >> ~/.bashrc && \
    echo "alias rl='ros2 launch'" >> ~/.bashrc && \
    echo "alias rt='ros2 topic'" >> ~/.bashrc && \
    echo "alias rn='ros2 node'" >> ~/.bashrc && \
    echo "alias rs='ros2 service'" >> ~/.bashrc

# ===== 권한 및 그룹 설정 =====
RUN usermod -a -G dialout root && \
    usermod -a -G tty root

# ===== 최종 정리 =====
RUN apt autoremove -y && \
    apt autoclean && \
    rm -rf /var/lib/apt/lists/*

# ===== 작업 디렉토리 권한 설정 =====
RUN chmod 755 /workspace
```

이제 정말 거의 모든 것이 포함되어 있어요:

- **모든 ROS2 Control 관련 패키지**
- **MoveIt 완전 세트**
- **모든 주요 센서 드라이버들**
- **시뮬레이션 도구들**
- **네비게이션 & SLAM**
- **컴퓨터 비전 라이브러리들**
- **네트워크 통신 도구들**
- **개발 및 디버깅 도구들**
- **편의 기능들 (별명, 환경설정)**

용량은 좀 클 거예요 (4-5GB), 하지만 이걸로 거의 모든 로봇 프로젝트를 커버할 수 있을 거예요!