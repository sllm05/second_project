SHELL := /bin/bash
.DEFAULT_GOAL := help

# GPU 변수는 l4|t4|v100 중 하나로 지정
GPU ?=

.PHONY: help cuda l4 t4 v100 full-install deps pyenv venv all check-ubuntu

help:
	@echo "사용법:"
	@echo "  make cuda GPU=<l4|t4|v100>   # CUDA/드라이버 설치 (CUDA 12.2, 550-server 계열+유틸)"
	@echo "  make l4 | make t4 | make v100 # 개별 GPU용 설치 바로 실행"
	@echo "  make full-install GPU=<l4|t4|v100>  # 전체 설치 플로우 실행 (Ubuntu 22.04 권장)"
	@echo "  make deps    # pyenv 종속 패키지 설치"
	@echo "  make pyenv   # pyenv 설치/설정"
	@echo "  make venv    # python 가상환경 생성/활성"
	@echo "  make check-ubuntu  # Ubuntu 22.04 권장 여부 확인"
	@echo "  make all GPU=<l4|t4|v100>  # deps -> pyenv -> venv 실행"

check-ubuntu:
	@UBU_VER=$$(lsb_release -rs 2>/dev/null || echo ""); \
	if [ -z "$$UBU_VER" ]; then \
	  echo "[안내] Ubuntu 22.04 (Jammy) 환경에서 검증됨. 다른 버전은 실패 가능"; \
	elif [ "$$UBU_VER" != "22.04" ]; then \
	  echo "[경고] 현재 Ubuntu $$UBU_VER. Ubuntu 22.04 (Jammy) 사용 권장"; \
	else \
	  echo "[확인] Ubuntu 22.04 (Jammy)"; \
	fi

cuda:
	@if [ -z "$(GPU)" ]; then \
	  echo "오류: GPU 변수를 지정하세요. 예) make cuda GPU=l4"; \
	  exit 1; \
	fi
	@chmod +x cuda_install.sh || true
	@bash cuda_install.sh $(GPU)

l4:
	@$(MAKE) cuda GPU=l4

t4:
	@$(MAKE) cuda GPU=t4

v100:
	@$(MAKE) cuda GPU=v100

full-install:
	@if [ -z "$(GPU)" ]; then \
	  echo "오류: GPU 변수를 지정하세요. 예) make full-install GPU=l4"; \
	  exit 1; \
	fi
	@chmod +x full_install.sh || true
	@bash full_install.sh $(GPU)

deps:
	@chmod +x dependencies_install.sh || true
	@bash dependencies_install.sh

pyenv:
	@chmod +x pyenv_setup.sh || true
	@bash pyenv_setup.sh

venv:
	@chmod +x python_virtualenv.sh || true
	@bash python_virtualenv.sh

all: deps pyenv venv
	@echo "기본 개발환경 구성이 완료되었습니다. 필요 시 make full-install GPU=<type> 실행을 진행하세요."


