# 법률 검색 플랫폼 - 웹 인터페이스

AI 기반 법률/매뉴얼 검색 및 질의응답 시스템의 프론트엔드 웹 인터페이스입니다.

## ✨ 주요 기능

- 🔍 **AI 기반 카테고리 추천**: 질문을 입력하면 관련도 높은 법률/매뉴얼 자동 추천
- 🎯 **다중 카테고리 검색**: 최대 5개 카테고리 동시 검색 가능
- 🤖 **LLM 답변 생성**: VARCO-VISION 모델 기반 정확한 답변 제공
- 📚 **참조 문서 제공**: 답변 근거가 된 법률 조문/매뉴얼 내용 확인
- 👍👎 **피드백 시스템**: 사용자 평가를 통한 시스템 품질 향상

---

## 🚀 빠른 시작

### 1. 백엔드 서버 실행 (필수!)

프론트엔드를 실행하기 전에 **반드시** 백엔드 API 서버를 먼저 실행해야 합니다.

```bash
# 프로젝트 루트에서
cd 03_api
python start_services.py
```

**서버가 정상 실행되면:**
- ✅ FastAPI: http://localhost:8000
- ✅ 임베딩 서버: http://localhost:8401  
- ✅ vLLM 서버: http://localhost:8400

자세한 내용은 [../03_api/API_readme.md](../03_api/API_readme.md) 참고

---

### 2. 프론트엔드 실행

#### **방법 1: 브라우저로 직접 열기** (가장 간단)

```bash
# 04_web 폴더에서 index.html을 더블클릭
# 또는
open index.html  # Mac
start index.html # Windows
```

⚠️ **주의**: `file://` 프로토콜로 열면 CORS 에러가 날 수 있습니다. 이 경우 방법 2 사용

---

#### **방법 2: Python HTTP 서버 사용** (권장)

```bash
cd 04_web
python -m http.server 3000
```

그 다음 브라우저에서 접속:
- 🌐 http://localhost:3000

---

#### **방법 3: VS Code Live Server**

1. VS Code에서 `04_web/index.html` 열기
2. 우클릭 → "Open with Live Server"
3. 자동으로 브라우저가 열립니다

---

#### **방법 4: Node.js http-server** (선택)

```bash
# 전역 설치 (최초 1회)
npm install -g http-server

# 실행
cd 04_web
http-server -p 3000 -c-1
```

---

## 📖 사용 방법

### Step 1: 질문 입력
1. **검색 범위 선택**: 법령 / 매뉴얼 / 전체
2. **질문 입력**: 구체적일수록 정확한 답변
3. **"연관 자료 추천" 클릭**

```
예시 질문:
- HACCP 인증 절차는 어떻게 되나요?
- 식품위생법 제48조의 내용은?
- 수출 식품의 위생 기준은 무엇인가요?
```

### Step 2: 카테고리 선택
- 추천된 카테고리 중 **최대 5개** 선택
- 관련도 점수(⭐) 참고
- **"검색 시작" 클릭**

### Step 3: 답변 확인
- AI가 생성한 답변 확인
- 검색 시간 / 생성 시간 표시
- 법률 자문 아님을 명시

### Step 4: 참조 문서 & 피드백
- **"자세히 보기"**: 법률 조문 전체 내용
- **👍 도움됨 / 👎 도움안됨**: 각 문서 평가
- **"피드백 전송"**: 평가 결과 전송 (학습 데이터화)

---

## 🔧 트러블슈팅

### ❌ "API 서버에 연결할 수 없습니다"

**원인**: 백엔드 서버가 실행되지 않음

**해결**:
```bash
# 백엔드 서버 실행 확인
cd 03_api
python start_services.py

# 또는 직접 확인
curl http://localhost:8000/health
```

---

### ❌ CORS 에러

**원인**: `file://` 프로토콜로 HTML 파일을 직접 열었을 때

**해결**: Python HTTP 서버 사용 (방법 2)
```bash
cd 04_web
python -m http.server 3000
# http://localhost:3000 접속
```

---

### ❌ "답변 생성에 실패했습니다"

**가능한 원인**:
1. vLLM 서버 미실행
2. GPU 메모리 부족
3. 인덱스 파일 누락

**해결**:
```bash
# 1. vLLM 서버 상태 확인
curl http://localhost:8400/v1/models

# 2. 로그 확인
cat 03_api/vllm_server.log

# 3. 인덱스 존재 확인
ls 00_data/input/indexes/law/2025-11-11/
ls 00_data/input/indexes/manual/2025-11-11/
```

---

### ❌ "카테고리 추천이 완료되었지만 아무것도 안 보임"

**원인**: 인덱스 파일에 문제가 있거나 검색 점수가 너무 낮음

**해결**:
1. 브라우저 콘솔(F12) 확인
2. 질문을 더 구체적으로 수정
3. 백엔드 로그 확인

---

## 🎨 커스터마이징

### API 서버 주소 변경

`script.js` 파일 상단에서 수정:

```javascript
// 기본값
const API_BASE_URL = 'http://localhost:8000';

// 프로덕션 서버로 변경 시
const API_BASE_URL = 'https://your-api-server.com';
```

---

### 디자인 색상 변경

`style.css` 파일 상단 CSS 변수 수정:

```css
:root {
    --primary-blue: #2563EB;    /* 메인 파란색 */
    --primary-purple: #7C3AED;  /* 메인 보라색 */
    /* ... */
}
```

---

## 📂 파일 구조

```
04_web/
├── index.html          # HTML 구조 (시맨틱 마크업)
├── style.css           # CSS 스타일 (글래스모피즘, 애니메이션)
├── script.js           # JavaScript 로직 (API 연동, 이벤트 처리)
└── README.md           # 이 문서
```

**총 용량**: ~50KB (경량!)

---

## 🔗 API 엔드포인트

프론트엔드에서 사용하는 API 엔드포인트:

| 단계 | 메서드 | 엔드포인트 | 설명 |
|------|--------|-----------|------|
| 1 | POST | `/queries` | 질문 → 카테고리 추천 |
| 2 | POST | `/answers` | 카테고리 → 답변 생성 |
| 3 | GET | `/answers/{answer_id}/chunks/{chunk_id}` | 청크 상세 조회 |
| 4 | POST | `/feedback/chunks` | 피드백 전송 |

자세한 API 스펙: [../03_api/API_readme.md](../03_api/API_readme.md)

---

## 🌟 기술 스택

### 프론트엔드
- **Vanilla JavaScript** (ES6+) - 프레임워크 없음
- **HTML5** - 시맨틱 마크업
- **CSS3** - 글래스모피즘, 그라데이션, 애니메이션
- **Google Fonts** (Noto Sans KR)

### 백엔드 (연동)
- **FastAPI** - REST API
- **RAG 엔진** - 검색 증강 생성
- **vLLM** - LLM 추론 서버

---

## 📊 브라우저 지원

| 브라우저 | 최소 버전 |
|---------|----------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

---

## 🚢 프로덕션 배포 가이드

### 1. 정적 호스팅 (Netlify, Vercel)

```bash
# 04_web 폴더를 업로드
netlify deploy --dir=04_web --prod
```

**주의**: `script.js`에서 API_BASE_URL을 프로덕션 서버로 변경!

---

### 2. Nginx 서버

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /path/to/04_web;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API 프록시 (CORS 우회)
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

### 3. Docker

```dockerfile
FROM nginx:alpine
COPY 04_web /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 🤝 기여 가이드

1. 버그 발견 시 이슈 등록
2. 기능 개선 제안 환영
3. Pull Request 작성 시 테스트 필수

---

## 📝 라이선스

이 프로젝트는 연구 목적으로 제작되었습니다.

---

## 📞 문의

- 기술 문의: 프로젝트 이슈 등록
- API 관련: [../03_api/API_readme.md](../03_api/API_readme.md) 참고
- RAG 엔진: [../02_rag/](../02_rag/) 참고

---

## 🎉 완료!

이제 http://localhost:3000 에서 법률 검색 플랫폼을 사용할 수 있습니다!

**문제가 생기면?**
1. 브라우저 콘솔(F12) 확인
2. 백엔드 로그 확인
3. 이 README의 트러블슈팅 섹션 참고

