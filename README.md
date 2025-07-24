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
