// ============================================
// 설정
// ============================================
// 자동으로 현재 호스트의 API 서버 연결
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'  // 로컬 개발
    : `http://${window.location.hostname}:8000`;  // 회사 서버

// ============================================
// 전역 상태
// ============================================
let state = {
    queryId: null,
    answerId: null,
    question: '',
    scope: 'all',
    categories: [],
    selectedCategories: [],
    citations: [],
    feedbackData: {} // { chunk_id: 'positive' | 'negative' }
};

// ============================================
// DOM 요소
// ============================================
const elements = {
    // Step 1
    questionInput: document.getElementById('question'),
    scopeAll: document.getElementById('scope-all'),
    scopeLaw: document.getElementById('scope-law'),
    scopeManual: document.getElementById('scope-manual'),
    btnRecommend: document.getElementById('btn-recommend'),
    
    // Step 2
    sectionCategories: document.getElementById('section-categories'),
    categoryList: document.getElementById('category-list'),
    btnSearch: document.getElementById('btn-search'),
    
    // Loading
    sectionLoading: document.getElementById('section-loading'),
    
    // Step 3
    sectionAnswer: document.getElementById('section-answer'),
    answerText: document.getElementById('answer-text'),
    timingRetrieval: document.getElementById('timing-retrieval'),
    timingGeneration: document.getElementById('timing-generation'),
    
    // Step 4
    sectionCitations: document.getElementById('section-citations'),
    citationsList: document.getElementById('citations-list'),
    btnSubmitFeedback: document.getElementById('btn-submit-feedback'),
    
    // Modal
    modal: document.getElementById('modal-chunk'),
    modalChunkText: document.getElementById('modal-chunk-text'),
    modalClose: document.querySelector('.modal-close'),
    modalOverlay: document.querySelector('.modal-overlay'),
    
    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.querySelector('.toast-message')
};

// ============================================
// 이벤트 리스너 등록
// ============================================
function initEventListeners() {
    // Step 1: 연관 자료 추천
    elements.btnRecommend.addEventListener('click', handleRecommendCategories);
    
    // Step 2: 검색 시작
    elements.btnSearch.addEventListener('click', handleSearch);
    
    // Step 4: 피드백 전송
    elements.btnSubmitFeedback.addEventListener('click', handleSubmitFeedback);
    
    // 모달 닫기
    elements.modalClose.addEventListener('click', closeModal);
    elements.modalOverlay.addEventListener('click', closeModal);
    
    // Enter 키로 질문 제출
    elements.questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleRecommendCategories();
        }
    });
}

// ============================================
// Step 1: 연관 자료 추천
// ============================================
async function handleRecommendCategories() {
    try {
        // 입력 검증
        const question = elements.questionInput.value.trim();
        if (!question) {
            showToast('질문을 입력해주세요', 'error');
            elements.questionInput.focus();
            return;
        }
        
        // 범위 가져오기
        const scope = document.querySelector('input[name="scope"]:checked').value;
        
        // 버튼 비활성화
        elements.btnRecommend.disabled = true;
        elements.btnRecommend.innerHTML = `
            <svg class="spinner" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="2" fill="none" opacity="0.3"/>
            </svg>
            <span>추천 중...</span>
        `;
        
        // API 호출
        const response = await fetch(`${API_BASE_URL}/queries`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, scope })
        });
        
        if (!response.ok) {
            throw new Error(`API 오류: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 상태 업데이트
        state.queryId = data.query_id;
        state.question = question;
        state.scope = scope;
        state.categories = data.category_candidates || [];
        state.selectedCategories = [];
        
        // UI 업데이트
        renderCategories();
        showSection('categories');
        
        // 스크롤
        elements.sectionCategories.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        showToast('카테고리 추천이 완료되었습니다', 'success');
        
    } catch (error) {
        console.error('카테고리 추천 실패:', error);
        showToast('카테고리 추천에 실패했습니다. 다시 시도해주세요.', 'error');
    } finally {
        // 버튼 복원
        elements.btnRecommend.disabled = false;
        elements.btnRecommend.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"/>
            </svg>
            <span>연관 자료 추천</span>
        `;
    }
}

// ============================================
// 카테고리 렌더링
// ============================================
function renderCategories() {
    elements.categoryList.innerHTML = '';
    
    if (state.categories.length === 0) {
        elements.categoryList.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                추천 가능한 카테고리가 없습니다.
            </div>
        `;
        return;
    }
    
    state.categories.forEach((category, index) => {
        const categoryEl = document.createElement('div');
        categoryEl.className = 'category-item';
        categoryEl.dataset.categoryId = category.category_id;
        
        categoryEl.innerHTML = `
            <input type="checkbox" class="category-checkbox" id="cat-${index}">
            <div class="category-content">
                <div class="category-header">
                    <label for="cat-${index}" class="category-name">${category.label}</label>
                    <div class="category-check">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
                        </svg>
                    </div>
                </div>
                ${category.score !== undefined && category.score !== 1.0 ? 
                    `<div class="category-score">${category.score.toFixed(2)}</div>` : 
                    ''}
            </div>
        `;
        
        // 클릭 이벤트
        categoryEl.addEventListener('click', () => toggleCategory(category.category_id));
        
        elements.categoryList.appendChild(categoryEl);
    });
    
    // 첫 번째 카테고리 자동 선택
    if (state.categories.length > 0) {
        toggleCategory(state.categories[0].category_id);
    }
}

// ============================================
// 카테고리 선택/해제
// ============================================
function toggleCategory(categoryId) {
    const categoryEl = document.querySelector(`[data-category-id="${categoryId}"]`);
    const checkbox = categoryEl.querySelector('.category-checkbox');
    
    // 토글
    checkbox.checked = !checkbox.checked;
    
    if (checkbox.checked) {
        // 최대 5개 제한
        if (state.selectedCategories.length >= 5) {
            showToast('최대 5개까지 선택할 수 있습니다', 'error');
            checkbox.checked = false;
            return;
        }
        categoryEl.classList.add('selected');
        state.selectedCategories.push(categoryId);
    } else {
        categoryEl.classList.remove('selected');
        state.selectedCategories = state.selectedCategories.filter(id => id !== categoryId);
    }
    
    // 검색 버튼 활성화/비활성화
    elements.btnSearch.disabled = state.selectedCategories.length === 0;
}

// ============================================
// Step 2: 검색 시작 (답변 생성)
// ============================================
async function handleSearch() {
    try {
        if (state.selectedCategories.length === 0) {
            showToast('최소 1개 이상의 카테고리를 선택해주세요', 'error');
            return;
        }
        
        // 로딩 표시
        showSection('loading');
        elements.sectionLoading.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // API 호출
        const response = await fetch(`${API_BASE_URL}/answers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query_id: state.queryId,
                selected_categories: state.selectedCategories
            })
        });
        
        if (!response.ok) {
            throw new Error(`API 오류: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 상태 업데이트
        state.answerId = data.answer_id;
        state.citations = data.citations || [];
        state.feedbackData = {};
        
        // UI 업데이트
        renderAnswer(data);
        renderCitations(data.citations);
        
        // 섹션 표시
        hideSection('loading');
        showSection('answer');
        showSection('citations');
        
        // 스크롤
        elements.sectionAnswer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        showToast('답변이 생성되었습니다', 'success');
        
    } catch (error) {
        console.error('답변 생성 실패:', error);
        hideSection('loading');
        showToast('답변 생성에 실패했습니다. 다시 시도해주세요.', 'error');
    }
}

// ============================================
// 답변 렌더링
// ============================================
function renderAnswer(data) {
    // 답변 텍스트
    elements.answerText.textContent = data.answer?.text || '답변을 생성할 수 없습니다.';
    
    // 타이밍 정보
    const timings = data.timings || {};
    elements.timingRetrieval.textContent = timings.retrieval_ms 
        ? `🔍 검색: ${timings.retrieval_ms}ms` 
        : '';
    elements.timingGeneration.textContent = timings.generation_ms 
        ? `🤖 생성: ${timings.generation_ms}ms` 
        : '';
}

// ============================================
// 참조 문서 렌더링
// ============================================
function renderCitations(citations) {
    elements.citationsList.innerHTML = '';
    
    if (!citations || citations.length === 0) {
        elements.citationsList.innerHTML = `
            <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                참조할 문서가 없습니다.
            </div>
        `;
        elements.btnSubmitFeedback.disabled = true;
        return;
    }
    
    citations.forEach((citation, index) => {
        const citationEl = document.createElement('div');
        citationEl.className = 'citation-card';
        citationEl.dataset.chunkId = citation.chunk_id;
        
        citationEl.innerHTML = `
            <div class="citation-header">
                <div class="citation-title">${citation.doc_title || '제목 없음'}</div>
                <div class="citation-score">${citation.score ? citation.score.toFixed(2) : 'N/A'}</div>
            </div>
            <div class="citation-actions">
                <button class="citation-btn btn-detail" data-chunk-id="${citation.chunk_id}">
                    📄 자세히 보기
                </button>
                <button class="citation-btn btn-feedback" data-chunk-id="${citation.chunk_id}" data-type="positive">
                    👍 도움됨
                </button>
                <button class="citation-btn btn-feedback" data-chunk-id="${citation.chunk_id}" data-type="negative">
                    👎 도움안됨
                </button>
            </div>
        `;
        
        elements.citationsList.appendChild(citationEl);
    });
    
    // 이벤트 리스너 등록
    document.querySelectorAll('.btn-detail').forEach(btn => {
        btn.addEventListener('click', () => handleViewDetail(btn.dataset.chunkId));
    });
    
    document.querySelectorAll('.btn-feedback').forEach(btn => {
        btn.addEventListener('click', () => handleFeedbackClick(btn));
    });
    
    elements.btnSubmitFeedback.disabled = false;
}

// ============================================
// Step 3: 자세히 보기
// ============================================
async function handleViewDetail(chunkId) {
    try {
        // API 호출
        const response = await fetch(
            `${API_BASE_URL}/answers/${state.answerId}/chunks/${chunkId}`
        );
        
        if (!response.ok) {
            throw new Error(`API 오류: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 모달 표시
        elements.modalChunkText.textContent = data.chunk_text || '내용을 불러올 수 없습니다.';
        openModal();
        
    } catch (error) {
        console.error('상세 내용 조회 실패:', error);
        showToast('상세 내용을 불러올 수 없습니다', 'error');
    }
}

// ============================================
// 피드백 버튼 클릭
// ============================================
function handleFeedbackClick(button) {
    const chunkId = button.dataset.chunkId;
    const type = button.dataset.type; // 'positive' | 'negative'
    
    const card = button.closest('.citation-card');
    const allFeedbackBtns = card.querySelectorAll('.btn-feedback');
    
    // 같은 버튼 다시 클릭 시 취소
    if (state.feedbackData[chunkId] === type) {
        delete state.feedbackData[chunkId];
        allFeedbackBtns.forEach(btn => {
            btn.classList.remove('active-positive', 'active-negative');
        });
    } else {
        // 새로운 피드백 설정
        state.feedbackData[chunkId] = type;
        
        // UI 업데이트
        allFeedbackBtns.forEach(btn => {
            btn.classList.remove('active-positive', 'active-negative');
            if (btn.dataset.chunkId === chunkId && btn.dataset.type === type) {
                btn.classList.add(type === 'positive' ? 'active-positive' : 'active-negative');
            }
        });
    }
    
    // 피드백 전송 버튼 활성화/비활성화
    elements.btnSubmitFeedback.disabled = Object.keys(state.feedbackData).length === 0;
}

// ============================================
// Step 4: 피드백 전송
// ============================================
async function handleSubmitFeedback() {
    try {
        if (Object.keys(state.feedbackData).length === 0) {
            showToast('평가할 문서를 선택해주세요', 'error');
            return;
        }
        
        // 피드백 배열 생성
        const feedback = Object.entries(state.feedbackData).map(([chunkId, type]) => ({
            chunk_id: chunkId,
            feedback: type
        }));
        
        // 버튼 비활성화
        elements.btnSubmitFeedback.disabled = true;
        elements.btnSubmitFeedback.innerHTML = `
            <svg class="spinner" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="2" fill="none" opacity="0.3"/>
            </svg>
            <span>전송 중...</span>
        `;
        
        // API 호출
        const response = await fetch(`${API_BASE_URL}/feedback/chunks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                answer_id: state.answerId,
                query_id: state.queryId,
                feedback: feedback,
                meta: {
                    timestamp: new Date().toISOString(),
                    user_agent: navigator.userAgent
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`API 오류: ${response.status}`);
        }
        
        showToast('피드백이 전송되었습니다. 감사합니다!', 'success');
        
        // 피드백 버튼 비활성화 (중복 전송 방지)
        document.querySelectorAll('.btn-feedback').forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
        });
        
    } catch (error) {
        console.error('피드백 전송 실패:', error);
        showToast('피드백 전송에 실패했습니다', 'error');
        elements.btnSubmitFeedback.disabled = false;
    } finally {
        // 버튼 복원
        elements.btnSubmitFeedback.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
            </svg>
            <span>피드백 전송</span>
        `;
    }
}

// ============================================
// UI 유틸리티
// ============================================
function showSection(sectionName) {
    const sectionMap = {
        'categories': elements.sectionCategories,
        'loading': elements.sectionLoading,
        'answer': elements.sectionAnswer,
        'citations': elements.sectionCitations
    };
    
    const section = sectionMap[sectionName];
    if (section) {
        section.classList.remove('hidden');
    }
}

function hideSection(sectionName) {
    const sectionMap = {
        'categories': elements.sectionCategories,
        'loading': elements.sectionLoading,
        'answer': elements.sectionAnswer,
        'citations': elements.sectionCitations
    };
    
    const section = sectionMap[sectionName];
    if (section) {
        section.classList.add('hidden');
    }
}

// ============================================
// 모달
// ============================================
function openModal() {
    elements.modal.classList.add('show');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    elements.modal.classList.remove('show');
    document.body.style.overflow = '';
}

// ============================================
// Toast 알림
// ============================================
function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;
    elements.toast.className = `toast ${type}`;
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// ============================================
// 초기화
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    console.log('✅ 법률 검색 플랫폼이 준비되었습니다');
    console.log(`📡 API 서버: ${API_BASE_URL}`);
});

// ============================================
// API 연결 테스트 (선택)
// ============================================
async function testApiConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ API 서버 연결 성공');
            return true;
        }
    } catch (error) {
        console.error('❌ API 서버 연결 실패:', error);
        showToast('API 서버에 연결할 수 없습니다. 서버를 실행해주세요.', 'error');
        return false;
    }
}

// 페이지 로드 시 API 연결 테스트
window.addEventListener('load', testApiConnection);

