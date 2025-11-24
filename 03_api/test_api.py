"""
API 테스트 스크립트 (4단계 엑셀 요구사항 기준)

서버 실행 후 다른 터미널에서 실행:
python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def print_json(title, data):
    """JSON 데이터를 보기 좋게 출력"""
    print(f"\n{title}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_4step_api():
    """4단계 엑셀 요구사항 API 전체 테스트"""
    print("\n" + "="*80)
    print("📋 4단계 API 전체 테스트 (엑셀 요구사항 기준)")
    print("="*80)
    
    # ===== 1단계: POST /queries - 질문 → 카테고리 추천 =====
    print("\n" + "="*80)
    print("1️⃣  POST /queries - 질문 → 카테고리 추천")
    print("="*80)
    
    step1_request = {
        "question": "HACCP 인증 절차는 어떻게 되나요?",
        "scope": "all"
    }
    
    print_json("📤 Request:", step1_request)
    
    query_response = requests.post(
        f"{BASE_URL}/queries",
        json=step1_request
    )
    
    if query_response.status_code != 200:
        print(f"\n❌ 오류 발생: HTTP {query_response.status_code}")
        print(query_response.text)
        return
    
    query_data = query_response.json()
    print_json("📥 Response:", query_data)
    
    query_id = query_data['query_id']
    print(f"\n✅ 1단계 완료!")
    print(f"   - query_id: {query_id}")
    print(f"   - 추천 카테고리: {len(query_data['category_candidates'])}개 (전체 1개 + 추천 최대 2개)")
    
    # 카테고리 선택 시뮬레이션 (전체 제외하고 추천 카테고리만 선택)
    selected_cat_ids = [cat['category_id'] for cat in query_data['category_candidates'][1:3]]
    if not selected_cat_ids:
        # 추천이 없으면 전체 선택
        selected_cat_ids = [query_data['category_candidates'][0]['category_id']]
    
    print(f"\n📌 사용자 선택 시뮬레이션:")
    print(f"   선택된 category_id: {selected_cat_ids}")
    
    # ===== 2단계: POST /answers - 답변 생성 =====
    print("\n" + "="*80)
    print("2️⃣  POST /answers - 답변 생성 (동기 - 5~10초 소요)")
    print("="*80)
    
    step2_request = {
        "query_id": query_id,
        "selected_categories": selected_cat_ids
    }
    
    print_json("📤 Request:", step2_request)
    print("\n⏳ 답변 생성 중... (검색 + LLM 생성)")
    
    answer_response = requests.post(
        f"{BASE_URL}/answers",
        json=step2_request
    )
    
    if answer_response.status_code != 200:
        print(f"\n❌ 오류 발생: HTTP {answer_response.status_code}")
        print(answer_response.text)
        return
    
    answer_data = answer_response.json()
    
    # Response 간소화 출력 (text는 너무 길어서 일부만)
    answer_data_display = answer_data.copy()
    if answer_data_display['answer']['text']:
        original_text = answer_data_display['answer']['text']
        answer_data_display['answer']['text'] = original_text[:100] + "..." if len(original_text) > 100 else original_text
    
    print_json("📥 Response:", answer_data_display)
    
    answer_id = answer_data['answer_id']
    print(f"\n✅ 2단계 완료!")
    print(f"   - answer_id: {answer_id}")
    print(f"   - status: {answer_data['status']}")
    print(f"   - 답변 길이: {len(answer_data['answer']['text'])}자")
    print(f"   - citations: {len(answer_data['citations'])}개")
    print(f"   - 검색 시간: {answer_data['timings']['retrieval_ms']}ms")
    print(f"   - 생성 시간: {answer_data['timings']['generation_ms']}ms")
    
    # Citations 상세 정보
    if answer_data['citations']:
        print(f"\n📄 Citations 목록:")
        for i, cit in enumerate(answer_data['citations'], 1):
            print(f"   {i}. chunk_id: {cit['chunk_id']}")
            print(f"      doc_title: {cit['doc_title']}")
            print(f"      score: {cit['score']}")
    
    # ===== 3단계: POST /feedback/chunks - 청크 평가 저장 =====
    print("\n" + "="*80)
    print("3️⃣  POST /feedback/chunks - 청크 평가 저장 (👍👎 평가)")
    print("="*80)
    
    if not answer_data['citations']:
        print("⚠️  Citations가 없어서 3단계 테스트 생략")
    else:
        # 여러 청크에 대한 평가 시뮬레이션 (👍 positive, 👎 negative)
        feedback_list = []
        for i, cit in enumerate(answer_data['citations'][:2]):  # 최대 2개만 평가
            feedback_list.append({
                "chunk_id": cit['chunk_id'],
                "feedback": "positive" if i == 0 else "negative"  # 첫 번째는 👍, 두 번째는 👎
            })
        
        step3_request = {
            "answer_id": answer_id,
            "query_id": query_id,
            "feedback": feedback_list,
            "meta": {
                "user_id": "test_user",
                "session_id": "test_session_001"
            }
        }
        
        print_json("📤 Request:", step3_request)
        
        feedback_response = requests.post(
            f"{BASE_URL}/feedback/chunks",
            json=step3_request
        )
        
        if feedback_response.status_code != 204:
            print(f"\n❌ 오류 발생: HTTP {feedback_response.status_code}")
            print(feedback_response.text)
            return
        
        print(f"\n✅ HTTP 204 No Content - 피드백 저장 완료!")
        print(f"\n✅ 3단계 완료!")
        print(f"   - 피드백 저장: {len(feedback_list)}개 청크")
        print(f"   - Triplet 로그: positive → 긍정 샘플, negative → 부정 샘플")
        print(f"   - 저장 위치: 00_data/output/training_data/triplets_group_bgem3.jsonl")
    
    # ===== 4단계: GET /answers/{answer_id}/chunks/{chunk_id} - 청크 상세 조회 =====
    print("\n" + "="*80)
    print("4️⃣  GET /answers/{answer_id}/chunks/{chunk_id} - 청크 상세 조회 ([자세히] 버튼)")
    print("="*80)
    
    if not answer_data['citations']:
        print("⚠️  Citations가 없어서 4단계 테스트 생략")
    else:
        first_chunk_id = answer_data['citations'][0]['chunk_id']
        
        print(f"📤 Request: GET /answers/{answer_id}/chunks/{first_chunk_id}")
        
        chunk_response = requests.get(
            f"{BASE_URL}/answers/{answer_id}/chunks/{first_chunk_id}"
        )
        
        if chunk_response.status_code != 200:
            print(f"\n❌ 오류 발생: HTTP {chunk_response.status_code}")
            print(chunk_response.text)
            return
        
        chunk_data = chunk_response.json()
        
        # 텍스트가 너무 길면 일부만 표시
        chunk_data_display = chunk_data.copy()
        if len(chunk_data_display['chunk_text']) > 200:
            chunk_data_display['chunk_text'] = chunk_data_display['chunk_text'][:200] + "..."
        
        print_json("📥 Response:", chunk_data_display)
        
        print(f"\n✅ 4단계 완료!")
        print(f"   - chunk_id: {chunk_data['chunk_id']}")
        print(f"   - 전체 텍스트 길이: {len(chunk_data['chunk_text'])}자")
        print(f"   - 용도: 팝업창에 전체 조문 표시")


def test_health():
    """체크"""
    print("\n" + "="*80)
    print("🏥 서버 체크")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        print("✅ 서버 정상 작동")
        print_json("📥 Response:", response.json())
    else:
        print(f"❌ 서버 오류: HTTP {response.status_code}")


if __name__ == "__main__":
    try:
        # 헬스 체크
        test_health()
        
        # 4단계 전체 API 테스트
        test_4step_api()
        
        print("\n" + "="*80)
        print("✅ 전체 테스트 완료!")
        print("="*80)
        print("\n💡 Swagger UI에서도 테스트 가능: http://localhost:8000/docs")
    
    except requests.exceptions.ConnectionError:
        print("\n" + "="*80)
        print("❌ 서버 연결 실패")
        print("="*80)
        print("\n서버를 먼저 실행하세요:")
        print("  cd 03_api")
        print("  python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

