#!/bin/bash

# CUDA 및 NVIDIA Driver 설치
echo "Starting CUDA and NVIDIA Driver installation..."
bash cuda_install.sh

# 재부팅 이후에 이 스크립트를 다시 실행해야 합니다.
# 시스템 재부팅이 완료되었으면 아래 부분부터 실행하세요.
read -p "Has the system rebooted after CUDA installation? (yes/no): " REBOOT_CONFIRM
if [ "$REBOOT_CONFIRM" != "yes" ]; then
    echo "Please reboot your system and rerun this script."
    exit 1
fi

# pyenv 종속성 설치
echo "Installing dependencies for pyenv..."
bash dependencies_install.sh

# pyenv 설치 및 설정
echo "Setting up pyenv..."
bash pyenv_setup.sh

# Python 버전 및 가상환경 설정
echo "Installing Python and setting up virtual environment..."
PYTHON_VERSION="3.11.8"
VENV_NAME="my_env"

# Python 설치 및 virtualenv 생성
pyenv install $PYTHON_VERSION && echo "Python version $PYTHON_VERSION installed."
pyenv shell $PYTHON_VERSION && echo "Python version set to $PYTHON_VERSION."
pyenv virtualenv $PYTHON_VERSION "$VENV_NAME" && echo "Virtual environment '$VENV_NAME' created."
pyenv activate "$VENV_NAME" && echo "Virtual environment '$VENV_NAME' activated."

# Git 리포지토리 클론 (환경 변수로 URL 받아오기)
if [ -z "$GIT_REPO_URL" ]; then
    echo "Error: Please set the GIT_REPO_URL environment variable to the repository you want to clone."
    exit 1
fi

echo "Installing curl, git, and vim, and cloning GitHub repository from $GIT_REPO_URL..."
sudo apt-get update && sudo apt-get install -y curl git vim && echo "curl, git, vim installed."
git clone $GIT_REPO_URL && echo "Repository cloned from $GIT_REPO_URL."

echo "All installations and configurations are complete. 🎉"
