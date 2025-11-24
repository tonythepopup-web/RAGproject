# main_router_boot_once.py — (UPDATED: use MANUAL_PATH / LAW_PATH)
# 목적: LLM/임베딩/인덱스를 "한 번만" 로딩하고, 변경된 두 엔진 스크립트를
#       파일 경로(MANUAL_PATH, LAW_PATH)로 직접 로드해 법률/매뉴얼 모드 전환 실행.
# 사용 방법:
#   1) 아래 MANUAL_PATH / LAW_PATH 를 실제 파일 경로로 설정하세요.
#   2) python main_router_boot_once.py 실행
#      → 1) 법률 RAG / 2) 매뉴얼 RAG 선택 → 질의
# 전제:
#   - 각 스크립트에는 아래 심볼이 존재합니다:
#       * 전역: PARENT_DIR, LLM_MODEL_ID, SAVE_LOG (선택), desired_categories/mapping 등
#       * 함수: load_llm, init_rag_from_saved, preprocess_query, retrieve_docs,
#               classify_category, parse_category_input, build_chatml_prompt,
#               generate_llm_response, interactive_label_group
#
# 이 라우터의 핵심 아이디어:
#   - "Boot Once": 프로그램 시작 시 공용 LLM과 두 RAG의 인덱스를 모두 로딩해 두고,
#     이후 질의 시에는 로딩 없이 즉시 검색/생성/라벨링 루틴을 수행합니다.
#   - "Path-based import": import 경로 관리가 복잡해지는 것을 피하기 위해, 각 엔진
#     파일을 절대/상대 경로로 직접 로드합니다(importlib.util 사용). 이 때 각 엔진의
#     main()은 실행되지 않도록 안전하게 모듈 객체만 생성합니다.

import os
import importlib.util
from pathlib import Path
from datetime import datetime

# ====== 사용자 설정: 엔진 파일 경로 ======
# - 아래 두 상수는 실제 파일 경로로 지정하세요.

# 상대 경로 설정 (현재 파일 기준)
SCRIPT_DIR = Path(__file__).resolve().parent
MANUAL_PATH = str(SCRIPT_DIR / "run_manual_final.py")  # 매뉴얼 RAG 엔진 파일 경로
LAW_PATH    = str(SCRIPT_DIR / "run_law_final.py")     # 법률   RAG 엔진 파일 경로

# ====== 유틸: 파일 경로에서 모듈 로드 ======
# 목적:
#   - 파이썬 표준 import 경로(sys.path)에 의존하지 않고, 주어진 파일 경로에서
#     모듈을 동적으로 로드합니다.
# 동작:
#   1) 파일 존재성 검사 → spec 생성
#   2) spec.loader.exec_module(mod)로 로드(이때 __name__은 파일명 기반 별칭)
# 사용처:
#   - GlobalBoot.__init__ 내에서 매뉴얼/법률 엔진 모듈을 각각 로드합니다.

def load_module_from_path(mod_name: str, file_path: str):
    p = Path(file_path)
    if not p.is_file():
        # 파일이 없으면 즉시 실패시켜 사용자가 경로를 점검하도록 합니다.
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        # 로더가 없으면 파이썬이 해당 파일을 모듈로 해석하지 못하는 상황입니다.
        raise ImportError(f"spec 생성 실패: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        # 실제 모듈 로딩 수행. __name__ != "__main__" 이므로 엔진의 main()은 실행되지 않습니다.
        spec.loader.exec_module(mod)  # __name__ != "__main__" → main() 미실행
    except Exception as e:
        # 로딩 중 예외가 발생하면 원인과 함께 안내합니다(의존 라이브러리/문법 오류 등).
        raise ImportError(f"모듈 로딩 실패: {file_path} → {e}")
    return mod

# ====== 공통 제어 유틸 ======
# 목적:
#   - CLI 어디에서든 즉시 메뉴 복귀/프로그램 종료를 처리하기 위한 경량 예외 및 헬퍼.
# 사용:
#   - _check_cmd()는 입력 문자열을 검사해 'exit'/'menu' 등을 만나면 예외를 발생시켜
#     상위 호출 스택에서 흐름 제어를 단순화합니다.
class MenuExit(Exception):
    pass

class ProgramExit(Exception):
    pass


def _check_cmd(s: str):
    """입력 문자열에 제어 명령이 포함되어 있는지 확인하고, 있으면 예외를 던져 흐름 제어.
    - s: 사용자가 입력한 원문 문자열(또는 None)
    - return: 원문 문자열을 트리밍/소문자화한 뒤 그대로 반환(제어 명령이 아니면)
    - raise ProgramExit: 'exit', 'quit', 'q', '/quit'
    - raise MenuExit:    'menu', '/menu', '/m'
    """
    t = (s or "").strip().lower()
    if t in {"exit", "quit", "q", "/quit"}:  # 어디서든 종료
        raise ProgramExit()
    if t in {"menu", "/menu", "/m"}:          # 어디서든 메뉴 복귀
        raise MenuExit()
    return s


# ====== 글로벌 부트스트랩(한 번만 로딩) ======
# 역할:
#   - 프로그램 전역에서 재사용할 리소스(LLM, 두 RAG의 인덱스/설정)를 한 번만 로딩합니다.
#   - 이후 ask_manual()/ask_law() 호출 시 추가 로딩 없이 즉시 검색·생성·라벨링 수행.
class GlobalBoot:
    """프로그램 시작 시 한 번만 모든 자원 로딩 → 두 모드에서 재사용"""
    def __init__(self):
        # 0) 공통 모델 최우선 로드 (모든 초기화보다 먼저)
        #    - KeyBERT, bge-m3 임베딩 모델을 한 번만 로드
        #    - 법률/매뉴얼 모듈에서 중복 로드 방지
        
        # 0-1) KeyBERT 로드
        print("🔑 [KeyBERT] 로딩 중...")
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
            print("✅ [KeyBERT] 로드 완료!")
        except Exception as e:
            print(f"❌ [KeyBERT] 로드 실패: {e}")
            self.kw_model = None
        
        # 0-2) bge-m3 임베딩 모델 로드 (로컬 또는 원격)
        print("📦 [BGE-m3] 임베딩 모델 설정 중...")
        try:
            # config에서 설정 가져오기
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import USE_REMOTE_EMBEDDING, EMBEDDING_SERVER_URL, EmbeddingModelWrapper
            
            if USE_REMOTE_EMBEDDING:
                # 원격 임베딩 서버 사용
                print(f"   원격 임베딩 서버 사용: {EMBEDDING_SERVER_URL}")
                self.embed_model = EmbeddingModelWrapper(local_model=None, use_remote=True)
                print("✅ [BGE-m3] 원격 서버 설정 완료!")
            else:
                # 로컬 임베딩 모델 로드
                from sentence_transformers import SentenceTransformer
                EMBED_MODEL_NAME = "dragonkue/BGE-m3-ko"
                local_model = SentenceTransformer(EMBED_MODEL_NAME)
                self.embed_model = EmbeddingModelWrapper(local_model=local_model, use_remote=False)
                print("✅ [BGE-m3] 로컬 모델 로드 완료!")
        except Exception as e:
            print(f"❌ [BGE-m3] 설정 실패: {e}")
            self.embed_model = None
        
        # 1) 엔진 모듈 로딩(파일 경로)
        #    - MANUAL_PATH/LAW_PATH가 비어있거나 잘못된 경우 즉시 실패시켜 사용자에게 알림.
        if not MANUAL_PATH or not LAW_PATH:
            raise RuntimeError("MANUAL_PATH / LAW_PATH 를 설정하세요 (환경변수 또는 상수).")
        print(f"[INFO] MANUAL_PATH: {MANUAL_PATH}")
        print(f"[INFO] LAW_PATH   : {LAW_PATH}")
        #   - 실제 모듈 객체 로딩(이때 두 모듈의 심볼을 self.RM/self.LR로 바인딩)
        self.RM = load_module_from_path("manual_mod", MANUAL_PATH)
        self.LR = load_module_from_path("law_mod", LAW_PATH)
        
        # 공통 모델 주입 (모듈 전역 변수 직접 설정)
        if self.kw_model is not None:
            # 모듈의 전역 변수로 설정 (중복 로드 방지)
            self.RM.kw_model = self.kw_model
            self.LR.kw_model = self.kw_model
            print(f"[INFO] KeyBERT 모델 주입 완료 (RM, LR)")
        
        if self.embed_model is not None:
            # 모듈의 전역 변수로 설정 (중복 로드 방지)
            self.RM.embed_model = self.embed_model
            self.LR.embed_model = self.embed_model
            print(f"[INFO] BGE-m3 임베딩 모델 주입 완료 (RM, LR)")

        # 2) LLM 프로세서 로드 (vLLM 서버 사용을 위해 모델은 로드하지 않음)
        #    - 실제 모델은 vLLM 서버에서 서빙되므로 프로세서(토크나이저)만 필요
        #    - 모델 로딩 시간 및 메모리 절약
        self.llm_model = None  # vLLM 서버 사용으로 모델 직접 로드 불필요
        self.llm_processor = None
        self.use_llm = True  # vLLM 서버가 구동되어 있다고 가정
        
        try:
            # 프로세서만 로드 (RM 또는 LR 중 하나에서)
            _, self.llm_processor = self.RM.load_llm(self.RM.LLM_MODEL_ID)
            print(f"[INFO] LLM 프로세서 로드 완료: {self.RM.LLM_MODEL_ID}")
            print(f"[INFO] vLLM 서버 사용 (모델 직접 로드 생략)")
        except Exception as e_rm:
            print(f"[경고] RM.load_llm 실패: {e_rm}\n → LR.load_llm로 재시도")
            try:
                _, self.llm_processor = self.LR.load_llm(self.LR.LLM_MODEL_ID)
                print(f"[INFO] LLM 프로세서 로드 완료: {self.LR.LLM_MODEL_ID}")
                print(f"[INFO] vLLM 서버 사용 (모델 직접 로드 생략)")
            except Exception as e_lr:
                # 프로세서 로드 실패 시 use_llm을 False로 설정
                print(f"[경고] LLM 프로세서 로드 실패 → 검색/라벨링만 사용: {e_lr}")
                self.use_llm = False

        # 3) 매뉴얼 RAG 인덱스 일괄 로드 (디렉터리 스캔 기반)
        #    - RM.PARENT_DIR에서 카테고리를 스캔해 mapping/desired_categories 준비
        #    - 각 카테고리 및 전체 인덱스를 메모리에 로딩
        self.manual_parent = getattr(self.RM, "PARENT_DIR", None)
        if not self.manual_parent or not os.path.isdir(self.manual_parent):
            raise FileNotFoundError(f"[오류] 매뉴얼 PARENT_DIR 없음/비유효: {self.manual_parent}")
        self.manual_indices = self.RM.init_rag_from_saved(self.manual_parent)
        # 로그 파일 경로는 엔진의 SAVE_LOG가 있으면 사용, 없으면 기본 파일명으로 대체
        self.manual_log_path = getattr(self.RM, "SAVE_LOG", "chat_log.txt")
        # 로그 폴더 자동 생성
        Path(self.manual_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.manual_log_path, "a", encoding="utf-8") as log:
            log.write(f"\n\n===== MANUAL RAG 세션 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        print(f"[INFO] 매뉴얼 인덱스 로드 완료: {len(self.manual_indices)}개 (root={self.manual_parent})")

        # 4) 법률 RAG 인덱스 일괄 로드 (고정 카테고리 기반)
        #    - LR.PARENT_DIR에서 사전 정의된 desired_categories 기반으로 로드
        self.law_parent = getattr(self.LR, "PARENT_DIR", None)
        if not self.law_parent or not os.path.isdir(self.law_parent):
            raise FileNotFoundError(f"[오류] 법률 PARENT_DIR 없음/비유효: {self.law_parent}")
        self.law_indices = self.LR.init_rag_from_saved(self.law_parent)
        self.law_log_path = getattr(self.LR, "SAVE_LOG", "chat_log.txt")
        # 로그 폴더 자동 생성
        Path(self.law_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.law_log_path, "a", encoding="utf-8") as log:
            log.write(f"\n\n===== LAW RAG 세션 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        print(f"[INFO] 법률 인덱스 로드 완료: {len(self.law_indices)}개 (root={self.law_parent})")

    # -------- 매뉴얼 모드 1회 처리 --------
    # 입력:
    #   - q(str): 사용자가 입력한 질의
    # 동작:
    #   1) 카테고리 목록/추천 표시 → 사용자 선택 파싱
    #   2) retrieve_docs()로 top-5 검색(하이브리드) + 프롬프트용 상위 2개 선택
    #   3) LLM 사용 가능 시 답변 생성 및 로그 기록
    #   4) 라벨링 인터랙션으로 이동
    def ask_manual(self, q: str):
        RM = self.RM  # 가독성을 위해 지역 별칭 사용(원본 모듈 객체)
        # 카테고리 목록 출력 (RM.mapping은 0=전체)
        print("\n=== 전체 카테고리 목록(매뉴얼) ===")
        mapping = getattr(RM, "mapping", {})
        desired = getattr(RM, "desired_categories", [])
        for i in range(0, len(desired) + 1):
            name = mapping.get(str(i), "")
            print(f"{i}) {name}", end="    ")
            if (i % 6) == 5:
                print()
        print("\n")

        # 추천 카테고리(카테고리 임베딩이 준비되어 있을 때에만 동작)
        routed = RM.classify_category(q) if (desired and getattr(RM, "category_embeddings", None) is not None) else []
        if routed:
            print("=== 추천 카테고리 ===")
            for i, cat in enumerate(desired, start=1):
                if cat in routed:
                    print(f"{i}) {cat}")
        print("0) 전체")

        # 선택 입력 → 파싱(미입력 시 추천 1순위 → 전체)
        choice = input("번호/이름 입력(미입력=추천→전체/첫번째) >> ")
        _check_cmd(choice)  # 어디서든 'menu'/'exit' 처리
        sel = RM.parse_category_input((choice or "").strip()) if choice else None
        if not sel:
            sel = routed[0] if routed else "전체"
        if sel not in self.manual_indices or not self.manual_indices.get(sel):
            # 선택한 카테고리가 로드되지 않았을 때 안전하게 전체로 폴백
            print(f"[경고] '{sel}' 인덱스가 없어 '전체'로 대체합니다.")
            sel = "전체"

        # 검색 수행(하이브리드) — RM.retrieve_docs는 idx_dir 힌트를 통해 본문 주입 처리
        cfg = self.manual_indices.get(sel)
        uq = RM.preprocess_query(q)
        results_top5 = RM.retrieve_docs(
            uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"],
            alpha=0.9, top_k=5, idx_dir=cfg.get("idx_dir")
        )
        results_for_prompt = results_top5[:2]  # 매뉴얼 프롬프트는 상위 2개 블록 사용

        # 출처만 간단 표기(미리보기)
        print(f"\n=== {sel} 검색 결과 ===")
        for r in results_for_prompt:
            print(r.get("source", ""))

        # LLM 답변(가능 시)
        if self.use_llm and self.llm_model is not None:
            # ChatML 프롬프트 구성 후, 대화 포맷에 맞춰 generate 호출
            prompt = RM.build_chatml_prompt(q, results_for_prompt, max_blocks=2, wrap_width=80)
            conv = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            gen = RM.generate_llm_response(self.llm_model, self.llm_processor, conv, max_new_tokens=1024)
            print(f"\n✅ LLM 응답 (⏱ {gen['elapsed']:.2f}s)\n")
            print(gen["output"])  # 모델 출력 그대로 표시
            # 로그 파일에 상세 기록(질문/선택 카테고리/프롬프트/응답/시간)
            with open(self.manual_log_path, "a", encoding="utf-8") as log:
                log.write(f"\n👤 질문: {q}\n")
                log.write(f"📂 선택 카테고리: {sel}\n")
                for r in results_for_prompt:
                    log.write(f" - {r.get('source','')}\n")
                log.write("\n--- Rendered Prompt ---\n")
                log.write(gen["rendered_prompt"] + "\n")
                log.write(f"\n🤖 VARCO 응답:\n{gen['output']}\n")
                log.write(f"⏱ 소요시간: {gen['elapsed']:.2f}초\n")
        else:
            # LLM이 없을 때도 검색 결과만으로 라벨링 파이프라인은 정상 진행됩니다.
            print("\n[안내] LLM 비활성/로딩 실패로 답변 생성을 건너뜁니다.")

        # 라벨링 단계로 이동/명령 처리
        cmd = input("\n(엔터=라벨링, 'menu'=메뉴 복귀, 'exit'=종료) ")
        try:
            _check_cmd(cmd)
        except MenuExit:
            # 상위 run_mode 루프에서 메뉴로 복귀하도록 예외 전파
            raise
        except ProgramExit:
            # 즉시 종료 루틴으로 전파
            raise
        RM.interactive_label_group(q, results_top5, llm_used_n=len(results_for_prompt))

    # -------- 법률 모드 1회 처리 --------
    # 입력/동작은 ask_manual과 유사하나, 법령 데이터 특성에 맞춰 프롬프트 블록 상한(3) 사용.
    def ask_law(self, q: str):
        LR = self.LR  # 가독성용 지역 별칭
        print("\n=== 전체 카테고리 목록(법률) ===")
        for i in range(0, len(LR.desired_categories) + 1):
            print(f"{i}) {LR.mapping[str(i)]}", end="    ")
            if (i % 6) == 5:
                print()
        print("\n")

        # 질의 기반 추천 카테고리(있으면 표시)
        candidates_cat = LR.classify_category(q)
        print(" === 추천 카테고리  ===")
        for i, cat in enumerate(LR.desired_categories, start=1):
            if cat in candidates_cat:
                print(f"{i}) {cat}")
        print("0) 전체")

        # 선택 파싱 → 폴백(추천 1순위 또는 전체)
        choice = input("번호 입력(미입력=best추천) >> ")
        _check_cmd(choice)
        sel = LR.parse_category_input((choice or "").strip()) if choice else None
        if not sel:
            sel = candidates_cat[0] if candidates_cat else "전체"
        if sel not in self.law_indices or not self.law_indices.get(sel):
            print(f"[경고] '{sel}' 인덱스가 없어 '전체'로 대체합니다.")
            sel = "전체"

        # 검색 수행(법률 하이브리드) — 상위 5개 검색, 프롬프트에는 상위 3개 사용
        cfg = self.law_indices[sel]
        uq = LR.preprocess_query(q)
        results_top5 = LR.retrieve_docs(uq, cfg["model"], cfg["index"], cfg["docs"], cfg["chunks"], cfg["IDF"], top_k=5)
        results_for_prompt = results_top5[:3]

        # 간단 결과 표시
        print(f"\n=== {sel} 검색 결과 ===")
        for r in results_for_prompt:
            print(r["source"])  # 법률 쪽은 source 필드 존재 가정

        # LLM 답변(가능 시)
        if self.use_llm and self.llm_model is not None:
            chatml_prompt = LR.build_chatml_prompt(q, results_for_prompt, max_blocks=3, wrap_width=80)
            conversation = [{"role": "user", "content": [{"type": "text", "text": chatml_prompt}]}]
            gen = LR.generate_llm_response(self.llm_model, self.llm_processor, conversation, max_new_tokens=1024)
            print(f"\n✅ LLM 응답 (⏱ {gen['elapsed']:.2f}s)\n")
            print(gen["output"])
            with open(self.law_log_path, "a", encoding="utf-8") as log:
                log.write(f"\n👤 질문: {q}\n")
                log.write(f"📂 선택 카테고리: {sel}\n")
                for r in results_for_prompt:
                    log.write(f" - {r['source']}\n")
                log.write("\n--- Rendered Prompt ---\n")
                log.write(gen["rendered_prompt"] + "\n")
                log.write(f"\n🤖 VARCO 응답:\n{gen['output']}\n")
                log.write(f"⏱ 소요시간: {gen['elapsed']:.2f}초\n")
        else:
            print("\n[안내] LLM 비활성/로딩 실패로 답변 생성을 건너뜁니다.")

        # 라벨링 단계 진입 또는 메뉴/종료 제어
        cmd = input("\n(엔터=라벨링, 'menu'=메뉴 복귀, 'exit'=종료) ")
        try:
            _check_cmd(cmd)
        except MenuExit:
            raise
        except ProgramExit:
            raise
        LR.interactive_label_group(q, results_top5, llm_used_n=len(results_for_prompt))


# ====== 모드 실행 루프 ======
# 역할:
#   - 상위 메뉴에서 선택한 모드("법률"/"매뉴얼")에 따라 질문을 반복 입력받아
#     해당 모드의 ask_* 함수를 호출합니다. 'menu'/'exit' 제어는 _check_cmd가 담당.

def run_mode(mode: str, boot: GlobalBoot):
    print(f"\n[{mode}] 모드입니다. 메뉴로 돌아가려면 'menu' 또는 '/menu', 종료는 'exit'.")
    while True:
        q = input(f"\n[{mode}] 질문을 입력하세요 > ")
        try:
            _check_cmd(q)
        except MenuExit:
            print("메뉴로 돌아갑니다.")
            break
        except ProgramExit:
            print("종료합니다.")
            raise SystemExit(0)

        if not (q or "").strip():
            # 공백/빈 입력은 무시하고 다음 루프로 진행
            continue

        if mode == "매뉴얼":
            try:
                boot.ask_manual(q)
            except MenuExit:
                print("메뉴로 돌아갑니다.")
                break
            except ProgramExit:
                print("종료합니다.")
                raise SystemExit(0)
        else:  # 법률
            try:
                boot.ask_law(q)
            except MenuExit:
                print("메뉴로 돌아갑니다.")
                break
            except ProgramExit:
                print("종료합니다.")
                raise SystemExit(0)


# ====== 진입점 ======
# 역할:
#   - 프로그램 초기화(GlobalBoot) 후 텍스트 메뉴 루프를 돌면서 모드 전환을 처리합니다.
#   - try/except로 사용자 인터럽트(CTRL+C 등) 시 깔끔히 종료 메시지를 출력합니다.

def main():
    boot = GlobalBoot()  # ⬅️ 여기서 한 번만 모든 로딩 수행

    MENU = """
=========================
  RAG Router (Boot Once)
=========================
1) 법률 RAG
2) 매뉴얼 RAG
q) 종료
-------------------------
선택 > """.strip()

    while True:
        try:
            sel = input(MENU + " ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 입력 스트림 종료(EOF)나 키보드 인터럽트가 발생하면 종료합니다.
            print("\n종료합니다.")
            return

        if sel in ("q", "quit", "exit"):
            print("종료합니다.")
            return
        elif sel == "1":
            run_mode("법률", boot)
        elif sel == "2":
            run_mode("매뉴얼", boot)
        else:
            print("올바른 번호를 입력하세요 (1/2 또는 q).")


if __name__ == "__main__":
    # 파이썬 스크립트가 직접 실행될 때만 main() 진입.
    main()
