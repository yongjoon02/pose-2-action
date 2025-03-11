import sys
import os

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 프로젝트 루트 디렉토리 설정 및 sys.path에 추가
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# 필요한 모듈 임포트
from src.yolo_pose import main

if __name__ == "__main__":
    # 커맨드라인 인자를 그대로 main() 함수에 전달
    main()
