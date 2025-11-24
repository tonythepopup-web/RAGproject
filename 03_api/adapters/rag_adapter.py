"""
기존 RAG 시스템(haccp_rag.py)을 API에 연결하는 어댑터

Boot Once 패턴:
- 앱 시작 시 GlobalBoot 인스턴스를 1회만 생성
- 이후 모든 요청은 이 인스턴스를 재사용
"""
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

# 02_rag 경로 추가
RAG_DIR = Path(__file__).resolve().parent.parent.parent / "02_rag"
sys.path.insert(0, str(RAG_DIR))

# 기존 RAG 모듈 import
try:
    from haccp_rag import GlobalBoot, load_module_from_path, MANUAL_PATH, LAW_PATH
except ImportError as e:
    raise ImportError(
        f"02_rag/haccp_rag.py를 import할 수 없습니다: {e}\n"
        "경로를 확인하세요."
    )


class RAGAdapter:
    """
    기존 RAG 시스템을 API용으로 래핑
    
    주요 메서드:
    - get_recommended_categories(scope, question) → 카테고리 추천
    - generate_answer(question, categories, scope) → 답변 생성
    """
    
    def __init__(self):
        """
        Boot Once: GlobalBoot 인스턴스를 1회만 생성
        - LLM, 임베딩 모델, 인덱스 모두 로드
        """
        print("🚀 [RAGAdapter] GlobalBoot 초기화 중...")
        try:
            self.boot = GlobalBoot()
            print("✅ [RAGAdapter] 초기화 완료!")
        except Exception as e:
            print(f"❌ [RAGAdapter] 초기화 실패: {e}")
            raise
    
    def get_recommended_categories(
        self, 
        scope: str, 
        question: str, 
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        질문에 대한 관련 카테고리 추천 (전체 1개 + 추천 2개 = 총 3개)
        
        Args:
            scope: 검색 범위 ('law', 'manual', 'all')
            question: 사용자 질문
            top_k: 추천 카테고리 수 (기본 2개, 전체 제외)
        
        Returns:
            [
                {"category_id": "LAW_전체", "label": "전체", "score": 1.0},
                {"category_id": "LAW_가축전염병예방법", "label": "가축전염병 예방법", "score": 0.94},
                ...
            ]
        """
        try:
            import re
            import numpy as np
            results = []
            
            # 1. "전체" 옵션 추가 (항상 첫 번째)
            if scope == "law":
                results.append({
                    "category_id": "LAW_전체",
                    "label": "전체",
                    "score": 1.0
                })
            elif scope == "manual":
                results.append({
                    "category_id": "MANUAL_전체",
                    "label": "전체",
                    "score": 1.0
                })
            else:  # scope == "all"
                results.append({
                    "category_id": "ALL_전체",
                    "label": "전체",
                    "score": 1.0
                })
            
            # 2. 추천 카테고리 수집 (RAG 검색 점수 기반)
            recommendations = []
            
            # 법률 카테고리 추천 (실제 문서 검색 점수 기반)
            if scope in ["law", "all"]:
                # 질문 전처리
                uq = self.boot.LR.preprocess_query(question)
                
                # 각 카테고리별로 실제 검색 수행
                for cat, cfg in self.boot.law_indices.items():
                    if cat == "전체":
                        continue  # 전체는 추천에서 제외
                    
                    try:
                        # 각 카테고리에서 top-3 검색
                        search_results = self.boot.LR.retrieve_docs(
                            uq, cfg["model"], cfg["index"], cfg["docs"], 
                            cfg["chunks"], cfg["IDF"], top_k=3
                        )
                        # 최고 점수 추출
                        best_score = max((r.get("score", 0.0) for r in search_results), default=0.0)
                        
                        if best_score > 0.0:
                            cat_id = cat.replace(" ", "")
                            recommendations.append({
                                "category_id": f"LAW_{cat_id}",
                                "label": cat,
                                "score": round(best_score, 2)
                            })
                    except Exception as e:
                        print(f"⚠️ [법률 카테고리 '{cat}' 검색 실패]: {e}")
                        continue
            
            # 매뉴얼 카테고리 추천 (실제 문서 검색 점수 기반)
            if scope in ["manual", "all"]:
                # 질문 전처리
                uq = self.boot.RM.preprocess_query(question)
                
                # 각 카테고리별로 실제 검색 수행
                for cat, cfg in self.boot.manual_indices.items():
                    if cat == "전체" or cat == "all":
                        continue  # 전체는 추천에서 제외
                    
                    try:
                        # 각 카테고리에서 top-3 검색
                        search_results = self.boot.RM.retrieve_docs(
                            uq, cfg["model"], cfg["index"], cfg["docs"], 
                            cfg["chunks"], cfg["IDF"], alpha=0.9, top_k=3,
                            idx_dir=cfg.get("idx_dir")
                        )
                        # 최고 점수 추출
                        best_score = max((r.get("score", 0.0) for r in search_results), default=0.0)
                        
                        if best_score > 0.0:
                            # "번호." 부분 제거
                            cat_clean = re.sub(r'^\d+\.\s*', '', cat).strip()
                            cat_id = cat_clean.replace(" ", "_")
                            recommendations.append({
                                "category_id": f"MANUAL_{cat_id}",
                                "label": cat_clean,
                                "score": round(best_score, 2)
                            })
                    except Exception as e:
                        print(f"⚠️ [매뉴얼 카테고리 '{cat}' 검색 실패]: {e}")
                        continue
            
            # 3. 추천 카테고리 정렬 및 상위 top_k개 선택
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            results.extend(recommendations[:top_k])
            
            # 최종: 전체(1개) + 추천(최대 2개) = 최대 3개
            return results
        
        except Exception as e:
            print(f"❌ [get_recommended_categories] 오류: {e}")
            # 오류 시에도 "전체" 옵션은 반환
            if scope == "law":
                return [{"category_id": "LAW_전체", "label": "전체", "score": 1.0}]
            elif scope == "manual":
                return [{"category_id": "MANUAL_전체", "label": "전체", "score": 1.0}]
            else:
                return [{"category_id": "ALL_전체", "label": "전체", "score": 1.0}]
    
    def generate_answer(
        self, 
        question: str, 
        selected_categories: List[str],
        scope: str = "all"
    ) -> Dict[str, Any]:
        """
        선택한 카테고리 기반 답변 생성
        
        Args:
            question: 사용자 질문
            selected_categories: 선택한 카테고리 목록 (예: ["법률_식품위생법", "매뉴얼_HACCP관리"])
            scope: 'law', 'manual', 'all'
        
        Returns:
            {
                "answer": "생성된 답변",
                "citations": [
                    {"category": "법률_식품위생법", "content": "...", "rank": 1},
                    ...
                ]
            }
        """
        try:
            import time
            all_results = []
            search_types = []  # 검색한 타입 추적 (law/manual)
            
            # ===== 1. 검색 시간 측정 =====
            retrieval_start = time.time()
            
            # 카테고리별로 검색 수행
            for cat_full in selected_categories:
                # "법률_식품위생법" → ("법률", "식품위생법")
                if "_" in cat_full:
                    cat_type, cat_name = cat_full.split("_", 1)
                else:
                    cat_type = scope
                    cat_name = cat_full
                
                if cat_type == "법률":
                    results = self._search_law(question, cat_name)
                    all_results.extend(results)
                    search_types.append("law")
                elif cat_type == "매뉴얼":
                    results = self._search_manual(question, cat_name)
                    all_results.extend(results)
                    search_types.append("manual")
            
            # 점수 기준으로 정렬 후 상위 5개 선택
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            top_results = all_results[:5]
            
            print(f"🔍 [generate_answer] 검색 결과: {len(all_results)}개, 상위 {len(top_results)}개 선택")
            
            retrieval_end = time.time()
            retrieval_ms = int((retrieval_end - retrieval_start) * 1000)  # 실제 검색 시간
            
            # 결과가 없으면 조기 반환
            if not all_results:
                return {
                    "answer": "검색 결과가 없습니다. 다른 카테고리를 선택해주세요.",
                    "citations": [],
                    "timings": {
                        "retrieval_ms": retrieval_ms,
                        "generation_ms": 0
                    }
                }
            
            # 우선 검색 타입 결정 (law 우선, 없으면 manual)
            primary_type = "law" if "law" in search_types else "manual"
            
            # ===== 2. LLM 생성 시간 측정 =====
            generation_start = time.time()
            
            # vLLM 서버 사용 (조건 분기 없이 항상 호출)
            answer = ""
            answer = self._generate_llm_answer(question, top_results, primary_type)
            answer = "LLM을 사용할 수 없습니다. 검색 결과만 반환합니다." if not answer else answer
            
            generation_end = time.time()
            generation_ms = int((generation_end - generation_start) * 1000)  # 실제 생성 시간
            
            # Citations 생성 (text 포함 - 4단계 청크 상세 조회에 필요)
            citations = []
            print(f"📝 [generate_answer] Citations 생성 시작: {len(top_results)}개 결과")
            for i, r in enumerate(top_results, 1):
                # doc_title 생성: source가 문서 제목
                doc_title = r.get("source", "알 수 없음")
                
                # text 추출 (RAG 엔진에서 이미 변환됨)
                # 법률: enriched_text (조문 번호 + 본문)
                # 매뉴얼: embedding_text (평탄화된 마크다운, 테이블 포함, \n 포함)
                text_content = r.get("text", "")
                
                # source 부분 제거 (매뉴얼의 경우 첫 줄에 source가 포함될 수 있음)
                if "\n\n" in text_content and text_content.startswith("source:"):
                    text_content = "\n\n".join(text_content.split("\n\n")[1:])
                
                citations.append({
                    "chunk_id": f"c_{uuid.uuid4().hex[:8]}",  # 청크 고유 ID 생성
                    "doc_title": doc_title,  # 문서 제목 (예: "식품위생법(법률) 제48조")
                    "score": r.get("score", 0.0),  # 관련도 점수
                    "text": text_content,  # 청크 전체 텍스트 (4단계 조회용, 평탄화됨)
                    "source": r.get("source", doc_title),  # 원본 출처
                    "category": r.get("category", "")  # 카테고리 정보
                })
            
            print(f"✅ [generate_answer] Citations 생성 완료: {len(citations)}개")
            
            return {
                "answer": answer,
                "citations": citations,
                "timings": {
                    "retrieval_ms": retrieval_ms,   # 실제 측정값
                    "generation_ms": generation_ms  # 실제 측정값
                }
            }
        
        except Exception as e:
            print(f"❌ [generate_answer] 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "citations": [],
                "timings": {
                    "retrieval_ms": 0,
                    "generation_ms": 0
                }
            }
    
    def _search_law(self, question: str, category: str) -> List[Dict[str, Any]]:
        """법률 검색"""
        try:
            LR = self.boot.LR
            
            # 카테고리 키 찾기
            matched_key = None
            if category in self.boot.law_indices:
                # 정확히 매칭
                matched_key = category
            else:
                # 언더스코어/공백 정규화 비교
                for key in self.boot.law_indices.keys():
                    if key.replace('_', ' ').replace(' ', '') == category.replace('_', ' ').replace(' ', ''):
                        matched_key = key
                        break
            
            # 매칭 실패 시 전체 검색
            if not matched_key:
                print(f"  ❌ [_search_law] 매칭 실패: '{category}' → '전체' 사용")
                print(f"     사용 가능한 키: {list(self.boot.law_indices.keys())}")
                matched_key = "전체"
            else:
                print(f"  ✅ [_search_law] 매칭 성공: '{category}' → '{matched_key}'")
            
            cfg = self.boot.law_indices[matched_key]
            uq = LR.preprocess_query(question)
            results_raw = LR.retrieve_docs(
                uq, cfg["model"], cfg["index"], cfg["docs"], 
                cfg["chunks"], cfg["IDF"], top_k=5
            )
            
            # 원본 chunk에 category와 source 정보 추가
            for r in results_raw:
                r["category"] = f"법률_{category}"
                if "source" not in r:
                    r["source"] = f"법률_{category}"  # source가 없으면 카테고리 이름 사용
            
            return results_raw
        except Exception as e:
            print(f"❌ [_search_law] 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_manual(self, question: str, category: str) -> List[Dict[str, Any]]:
        """매뉴얼 검색"""
        try:
            import re
            MR = self.boot.RM
            
            # 디버깅 로그 (필요시 주석 해제)
            # print(f"\n[DEBUG _search_manual] 입력: '{category}'")
            
            # 카테고리 키 찾기 (번호 없는 이름 → 원본 키)
            matched_key = None
            if category in self.boot.manual_indices:
                matched_key = category
            else:
                # 번호 제거된 이름으로 검색 (예: "HACCP 인증 따라하기" → "11. HACCP 인증 따라하기")
                for key in self.boot.manual_indices.keys():
                    clean_key = re.sub(r'^\d+\.\s*', '', key).strip()
                    # 언더스코어/공백 정규화 비교 (원본 파일명의 언더스코어 처리)
                    if clean_key.replace('_', ' ') == category.replace('_', ' ') or key == category:
                        matched_key = key
                        break
            
            # 매칭 실패 시 전체 검색
            if not matched_key:
                matched_key = "all"  # 매뉴얼은 idx_all
                print(f"⚠️ [_search_manual] 매칭 실패: '{category}' → 'all' 사용")
                print(f"   사용 가능한 키: {list(self.boot.manual_indices.keys())}")
            # else:
            #     print(f"✅ [_search_manual] 매칭 성공: '{category}' → '{matched_key}'")
            
            cfg = self.boot.manual_indices[matched_key]
            uq = MR.preprocess_query(question)
            results_raw = MR.retrieve_docs(
                uq, cfg["model"], cfg["index"], cfg["docs"], 
                cfg["chunks"], cfg["IDF"], 
                alpha=0.9, top_k=5, idx_dir=cfg.get("idx_dir")
            )
            
            # 원본 chunk에 category와 source 정보 추가
            for r in results_raw:
                r["category"] = f"매뉴얼_{category}"
                if "source" not in r:
                    r["source"] = f"매뉴얼_{category}"  # source가 없으면 카테고리 이름 사용
            
            return results_raw
        except Exception as e:
            print(f"❌ [_search_manual] 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _remove_markdown(self, text: str) -> str:
        """마크다운 형식 제거 및 요약 부분 제거"""
        import re
        
        # 요약 부분 제거 (요약: ~ 답변: 사이의 텍스트 제거)
        text = re.sub(r'(?i)^.*?요약\s*[:：].*?(?=답변\s*[:：])', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)^답변\s*[:：]\s*', '', text)
        
        # ** 굵게 제거
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # * 기울임 제거
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # ## 제목 제거
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # - 리스트 제거
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        # 1. 숫자 리스트 제거
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _generate_llm_answer(
        self, 
        question: str, 
        results: List[Dict[str, Any]], 
        primary_type: str
    ) -> str:
        """LLM 답변 생성
        
        Args:
            question: 사용자 질문
            results: 검색 결과 (원본 chunk 구조)
            primary_type: 'law' 또는 'manual'
        """
        try:
            # primary_type에 따라 적절한 모듈 선택
            if primary_type == "law":
                module = self.boot.LR
                max_blocks = 3
            else:  # manual
                module = self.boot.RM
                max_blocks = 2
            
            # ChatML 프롬프트 생성 (원본 chunk 구조 그대로 전달)
            prompt = module.build_chatml_prompt(
                question, 
                results[:max_blocks], 
                max_blocks=max_blocks, 
                wrap_width=80
            )
            
            # LLM 호출
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            gen = module.generate_llm_response(
                self.boot.llm_model, 
                self.boot.llm_processor, 
                conversation, 
                max_new_tokens=1024
            )
            
            # 마크다운 제거 (순수 텍스트만)
            answer_text = gen["output"]
            answer_text = self._remove_markdown(answer_text)
            
            return answer_text
        
        except Exception as e:
            print(f"❌ [_generate_llm_answer] 오류: {e}")
            import traceback
            traceback.print_exc()
            return f"LLM 답변 생성 중 오류 발생: {str(e)}"

