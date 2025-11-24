"""
서버 자동 기동 스크립트

실행 순서:
1. FastAPI 서버 백그라운드 기동
2. RAGAdapter 초기화 완료 대기 (✅ [RAGAdapter] 초기화 완료! 확인)
3. vLLM 서버 자동 기동 (VARCO 모델, port 8400)

사용법:
    python start_services.py

종료:
    Ctrl+C (모든 서버 정상 종료)
"""

import subprocess
import time
import sys
import signal
import os
from pathlib import Path

# 프로세스 관리
embedding_proc = None
fastapi_proc = None
vllm_proc = None

def signal_handler(sig, frame):
    """Ctrl+C 시 모든 서버 정상 종료"""
    print("\n\n🛑 종료 신호 수신. 모든 서버를 종료합니다...")
    
    if vllm_proc:
        print("  - vLLM 서버 종료 중...")
        vllm_proc.terminate()
        vllm_proc.wait(timeout=10)
    
    if fastapi_proc:
        print("  - FastAPI 서버 종료 중...")
        fastapi_proc.terminate()
        fastapi_proc.wait(timeout=10)
    
    if embedding_proc:
        print("  - 임베딩 서버 종료 중...")
        embedding_proc.terminate()
        embedding_proc.wait(timeout=10)
    
    print("✅ 모든 서버가 정상 종료되었습니다.")
    sys.exit(0)

def wait_for_rag_init(timeout=600):
    """
    RAGAdapter 초기화 완료 대기
    
    FastAPI 로그에서 "✅ [RAGAdapter] 초기화 완료!" 메시지 확인
    또는 /health 엔드포인트 체크
    
    Args:
        timeout: 최대 대기 시간(초)
    
    Returns:
        bool: 초기화 성공 여부
    """
    print("⏳ RAGAdapter 초기화 대기 중...")
    print("   (KeyBERT, bge-m3, 인덱스 로드 진행 중...)")
    
    start_time = time.time()
    
    # /health 엔드포인트 체크 방식
    while time.time() - start_time < timeout:
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ RAGAdapter 초기화 완료 확인!")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # 서버가 아직 준비되지 않음
            pass
        except ImportError:
            print("⚠️ requests 패키지가 없습니다. 시간 기반 대기로 전환합니다.")
            # requests 없으면 고정 시간 대기
            time.sleep(120)  # 2분 대기
            return True
        
        # 5초마다 체크
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # 30초마다 진행 상황 출력
            print(f"   ... {elapsed}초 경과 (최대 {timeout}초)")
    
    print(f"❌ 타임아웃: {timeout}초 내에 초기화 완료되지 않았습니다.")
    return False

def start_embedding_server():
    """임베딩 서버 기동 (TEI Docker)"""
    global embedding_proc
    
    print("\n🚀 [1/4] 임베딩 서버 기동 중...")
    print("   모델: BAAI/bge-m3")
    print("   포트: 8401")
    print("   ⚠️  Docker가 설치되어 있어야 합니다!")
    
    # Docker 컨테이너 실행
    cmd = [
        "docker", "run", "-d",
        "--name", "tei-bge-m3",
        "--gpus", "all",
        "-p", "8401:80",
        "--restart", "unless-stopped",
        "ghcr.io/huggingface/text-embeddings-inference:latest",
        "--model-id", "BAAI/bge-m3"
    ]
    
    try:
        # 기존 컨테이너 제거 (있으면)
        subprocess.run(["docker", "rm", "-f", "tei-bge-m3"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        
        # 새 컨테이너 시작
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ 임베딩 서버 시작 실패: {result.stderr}")
            print("\n💡 대안: 로컬 모드로 실행하려면:")
            print("   export USE_REMOTE_EMBEDDING=false")
            return False
        
        print(f"✅ 임베딩 서버 시작됨 (Container ID: {result.stdout[:12]})")
        print(f"   엔드포인트: http://localhost:8401")
        
        # 서버 준비 대기 (간단한 헬스 체크)
        print("   서버 준비 대기 중...")
        time.sleep(5)
        
        return True
    
    except FileNotFoundError:
        print("❌ Docker가 설치되어 있지 않습니다!")
        print("\n💡 대안: 로컬 모드로 실행하려면:")
        print("   export USE_REMOTE_EMBEDDING=false")
        return False
    except Exception as e:
        print(f"❌ 임베딩 서버 시작 중 오류: {e}")
        return False

def start_fastapi():
    """FastAPI 서버 백그라운드 기동"""
    global fastapi_proc
    
    print("\n🚀 [2/4] FastAPI 서버 기동 중...")
    print("   포트: 8000")
    print("   경로: 03_api/main.py")
    
    # 현재 디렉토리 확인 (03_api 내부에서 실행)
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # 로그를 파일에 저장
    log_file = open("fastapi_server.log", "w", encoding="utf-8")
    
    fastapi_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent  # 03_api 디렉토리
    )
    
    print(f"✅ FastAPI 서버 시작됨 (PID: {fastapi_proc.pid})")
    print(f"   로그: 03_api/fastapi_server.log")
    
    # 서버 시작 대기 (짧은 대기)
    time.sleep(3)
    
    if fastapi_proc.poll() is not None:
        print("❌ FastAPI 서버 시작 실패")
        return False
    
    return True

def start_vllm():
    """vLLM 서버 기동 (VARCO 모델)"""
    global vllm_proc
    
    print("\n🚀 [4/4] vLLM 서버 기동 중...")
    print("   모델: NCSOFT/VARCO-VISION-2.0-14B")
    print("   포트: 8400")
    print("   ⚠️  모델 다운로드 시 시간이 소요될 수 있습니다...")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "NCSOFT/VARCO-VISION-2.0-14B",
        "--host", "0.0.0.0",
        "--port", "8400",
        "--kv-cache-dtype", "auto",
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8"
    ]
    
    # 로그를 파일에 저장
    log_file = open("vllm_server.log", "w", encoding="utf-8")
    
    vllm_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    print(f"✅ vLLM 서버 시작됨 (PID: {vllm_proc.pid})")
    print(f"   로그: 03_api/vllm_server.log")
    print(f"   엔드포인트: http://localhost:8400/v1")
    
    return True

def main():
    """메인 실행 함수"""
    # Ctrl+C 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*70)
    print("  HACCP RAG 서버 자동 기동 스크립트")
    print("="*70)
    print()
    print("📋 실행 순서:")
    print("  1. 임베딩 서버 기동 (bge-m3, port 8401)")
    print("  2. FastAPI 서버 기동 (port 8000)")
    print("  3. RAGAdapter 초기화 대기 (KeyBERT, 인덱스)")
    print("  4. vLLM 서버 자동 기동 (VARCO, port 8400)")
    print()
    print("⚠️  종료: Ctrl+C")
    print("="*70)
    print()
    
    # 1. 임베딩 서버 시작
    if not start_embedding_server():
        print("\n⚠️  임베딩 서버 시작 실패. 로컬 모드로 전환합니다.")
        os.environ["USE_REMOTE_EMBEDDING"] = "false"
    
    # 2. FastAPI 서버 시작
    if not start_fastapi():
        print("\n❌ FastAPI 서버 시작 실패. 종료합니다.")
        return 1
    
    # 3. RAGAdapter 초기화 대기
    print("\n⏳ [3/4] RAGAdapter 초기화 대기 중...")
    if not wait_for_rag_init(timeout=600):
        print("\n⚠️  초기화 타임아웃. vLLM 서버 기동을 계속 진행합니다...")
    
    # 4. vLLM 서버 시작
    if not start_vllm():
        print("\n❌ vLLM 서버 시작 실패")
        return 1
    
    # 완료 메시지
    print("\n" + "="*70)
    print("✅ 모든 서버가 정상 기동되었습니다!")
    print("="*70)
    print()
    print("📊 서버 상태:")
    print(f"  - 임베딩:   http://localhost:8401      (Docker: tei-bge-m3)")
    print(f"  - FastAPI:  http://localhost:8000      (PID: {fastapi_proc.pid})")
    print(f"  - vLLM:     http://localhost:8400/v1   (PID: {vllm_proc.pid})")
    print()
    print("📖 API 문서:")
    print("  - Swagger:  http://localhost:8000/docs")
    print("  - ReDoc:    http://localhost:8000/redoc")
    print()
    print("📝 로그 파일:")
    print("  - 임베딩:   docker logs tei-bge-m3")
    print("  - FastAPI:  03_api/fastapi_server.log")
    print("  - vLLM:     03_api/vllm_server.log")
    print()
    print("⚠️  종료: Ctrl+C")
    print("="*70)
    print()
    
    # 프로세스 모니터링
    try:
        while True:
            # FastAPI 프로세스 체크
            if fastapi_proc.poll() is not None:
                print(f"\n❌ FastAPI 서버가 종료되었습니다 (종료 코드: {fastapi_proc.poll()})")
                break
            
            # vLLM 프로세스 체크
            if vllm_proc and vllm_proc.poll() is not None:
                print(f"\n❌ vLLM 서버가 종료되었습니다 (종료 코드: {vllm_proc.poll()})")
                break
            
            time.sleep(5)
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

