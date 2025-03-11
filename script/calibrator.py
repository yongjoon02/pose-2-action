#!/usr/bin/env python3
import os
import sys

# src 폴더를 파이썬 모듈 경로에 추가합니다.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

from calibration import main

if __name__ == "__main__":
    main()
