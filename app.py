import streamlit as st
import pandas as pd
import time
import requests
import re
import random
from io import BytesIO
from PIL import Image as PILImage
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder (Pro)",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .log-box {
        height: 300px;
        overflow-y: scroll;
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 상태 관리
# ---------------------------------------------------------
if 'processed_data' not in st.session_state: st.session_state.processed_data = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'logs' not in st.session_state: st.session_state.logs = []
if 'best_model_name' not in st.session_state: st.session_state.best_model_name = None

def add_log(msg):
    st.session_state.logs.append(msg)

# ---------------------------------------------------------
# 3. 핵심 함수
# ---------------------------------------------------------
def get_random_delay():
    # 봇 탐지 회피를 위한 랜덤 대기 (1.2초 ~ 2.5초)
    return random.uniform(1.2, 2.5)

def get_best_gemini_model():
    """모델 자동 선정 (Flash 우선)"""
    try:
        models = list(genai.list_models())
        candidates = []
        for m in models:
            name = m.name.lower()
            if 'gemini' in name and 'pro' not in name and 'generateContent' in m.supported_generation_methods:
                candidates.append(m.name)
        
        # 최신(숫자 큼) -> Flash 포함 순으로 정렬
        candidates.sort(key=lambda x: ('2.0' in x, 'flash' in x, x), reverse=True)
        
        if candidates: return candidates[0]
        return 'gemini-1.5-flash'
    except:
        return 'gemini-1.5-flash'

def safe_download_image(url):
    """이미지 다운로드 (타임아웃 10초)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        return img
    except:
        return None

def image_to_bytes(img):
    """엑셀용 이미지 바이트 변환 (비율 유지 리사이징)"""
    # 엑셀 셀 크기(약 130x130)에 맞게 썸네일 생성
    img.thumbnail((130, 130))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

def search_with_retry(query, max_retries=3):
    """검색 재시도 로직"""
    for attempt in range(max_retries):
        try:
            # 검색어에 'image'를 추가하거나 빼면서 시도
            q = query if attempt == 0 else query.replace(" product", "")
            results = DDGS().images(keywords=q, region="wt-wt", safesearch="off", max_results=20)
            return [r['image'] for r in results if 'image' in r]
        except: 
            time.sleep(2)
    return []

def verify_with_gemini(model_name, img, product_name):
    """
    [기준 대폭 완화]
    제품 사진처럼 보이면 무조건 YES를 하도록 유도
    """
    try:
        model = genai.GenerativeModel(model_name)
        
        # 프롬프트: '제품'이면 무조건 통과시켜라.
        prompt = f"""
        Does this image look like a commercial product, item, or device related to '{product_name}'?
        
        Rules:
        1. Answer YES if it shows ANY product.
        2. Answer YES even if it has some text or white background is missing.
        3. Answer NO only if it is an error message, a blank page, or map.
        
        Output only one word: YES or NO.
        """
        
        response = model.generate_content(
            [prompt, img],
            generation_config=GenerationConfig(max_output_tokens=10, temperature=0.1),
            request_options={'timeout': 10}
        )
        
        answer = response.text.strip().upper()
        
        if "YES" in answer:
            return True, f"✅ 합격"
        else:
            return False, f"⛔ AI 거절"
            
    except Exception as e:
        # 에러나면 그냥 통과시킴 (이미지 확보 우선)
        return True, "⚠️ 에러(자동통과)"

def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    rows = []
    
    # 데이터 프레임 준비
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['처리결과'] = item['status']
        rows.append(row_data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res = pd.DataFrame(rows)
        df_res.to_excel(writer, index=False, sheet_name='Result')
        
        wb = writer.book
        ws = writer.sheets['Result']
        
        # 행 높이 설정 (이미지가 들어갈 공간)
        ws.set_default_row(100)
        
        # 텍스트 줄바꿈 및 정렬
        fmt_text = wb.add_format({'text_wrap': True, 'valign': 'vcenter'})
        ws.set_column(0, len(original_columns), 15, fmt_text)

        # 이미지/링크 삽입
        # 기존 데이터 컬럼 + 1(처리결과) 다음부터 시작
        start_col = len(original_columns) + 1
        
        # 헤더 쓰기
        for i in range(target_count):
            ws.write(0, start_col + (i*2), f"이미지_{i+1}")
            ws.write(0, start_col + (i*2) + 1, f"링크_{i+1}")
            # 열 너비 조정 (이미지 칸은 넓게, 링크 칸은 좁게)
            ws.set_column(start_col + (i*2), start_col + (i*2), 18) # 이미지칸
            ws.set_column(start_col + (i*2) + 1, start_col + (i*2) + 1, 10, fmt_text) # 링크칸

        for i, item in enumerate(data_list):
            row_idx = i + 1
            
            # 각 이미지별로 반복
            for k in range(target_count):
                # k번째 이미지가 있는지 확인
                if k < len(item['images_data']):
                    img_bytes = item['images_data'][k]
                    url_link = item['image_urls'][k]
                    
                    # 1. 이미지 삽입
                    col_img = start_col + (k*2)
                    if img_bytes:
                        ws.insert_image(row_idx, col_img, "img.jpg", {
                            'image_data': img_bytes,
                            'x_scale': 1, 'y_scale': 1,
                            'object_position': 1 # 셀과 함께 이동 및 크기 변함
                        })
                    
                    # 2. 링크 삽입 (바로 옆 칸)
                    col_link = start_col + (k*2) + 1
                    ws.write_url(row_idx, col_link, url_link, string="[보기]")

    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("⚡ Smart-Image-Finder (Pro)")
st.caption("AI 검수 기준 완화 & 엑셀 링크 기능 추가")

st.sidebar.title("설정 & 로그")
use_ai_check = st.sidebar.checkbox("AI 검수 사용하기", value=True, help="체크 해제하면 무조건 다운로드합니다.")
log_placeholder = st.sidebar.empty()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = ""

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.sidebar.text_input("Google API Key 입력", type="password")

uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])

if uploaded_file and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    if not st.session_state.best_model_name:
        with st.spinner("최적의 모델 검색 중..."):
            st.session_state.best_model_name = get_best_gemini_model()
    
    st.info(f"🤖 모델: {st.session_state.best_model_name}")

    df = pd.read_excel(uploaded_file)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: col_brand = st.selectbox("제조사 열", df.columns, index=0)
    with c2: col_model = st.selectbox("모델명 열", df.columns, index=1 if len(df.columns)>1 else 0)
    with c3: target_count = st.number_input("필요 사진 수", 1, 5, 1)

    if st.button("🚀 작업 시작"):
        st.session_state.logs = []
        st.session_state.processed_data = [] 
        st.session_state.is_processing = True
        st.session_state.stop_requested = False
        st.rerun()

# ---------------------------------------------------------
# 5. 실행 로직
# ---------------------------------------------------------
if st.session_state.is_processing:
    
    if st.button("🛑 중단하고 저장하기"):
        st.session_state.stop_requested = True
    
    progress_bar = st.progress(0)
    status_box = st.empty()
    
    start_idx = len(st.session_state.processed_data)
    total_rows = len(df)
    
    for i in range(start_idx, total_rows):
        if st.session_state.stop_requested: break
            
        row = df.iloc[i]
        full_name = f"{row[col_brand]} {row[col_model]}"
        
        status_box.markdown(f"**[{i+1}/{total_rows}]** 처리 중: `{full_name}`")
        add_log(f"▶ [{i+1}] {full_name}")
        
        candidates = search_with_retry(f"{full_name} product")
        valid_images_bytes = []
        valid_image_urls = [] # 링크 저장용
        
        if candidates:
            for url in candidates[:15]:
                if len(valid_images_bytes) >= target_count: break
                
                pil_img = safe_download_image(url)
                
                if pil_img:
                    is_ok = True
                    reason = "AI 미사용"
                    
                    if use_ai_check:
                        is_ok, reason = verify_with_gemini(st.session_state.best_model_name, pil_img, full_name)
                    
                    if is_ok:
                        add_log(f"  {reason}")
                        img_bytes = image_to_bytes(pil_img)
                        
                        valid_images_bytes.append(img_bytes)
                        valid_image_urls.append(url) # URL도 같이 저장
                        
                        if use_ai_check: time.sleep(get_random_delay())
                    else:
                        add_log(f"  {reason}")
                else:
                    pass 

        msg = f"{len(valid_images_bytes)}장 확보"
        add_log(f"  🏁 결과: {msg}")
            
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images_bytes,
            'image_urls': valid_image_urls,
            'status': msg
        })
        
        log_text = "\n".join(st.session_state.logs[-30:])
        log_placeholder.code(log_text)
        progress_bar.progress((i + 1) / total_rows)
    
    st.session_state.is_processing = False
    st.success("작업 완료!")

if len(st.session_state.processed_data) > 0:
    if st.button("📥 엑셀 파일 다운로드 생성"):
        data = create_excel(st.session_state.processed_data, df.columns.tolist(), target_count)
        st.download_button("다운로드", data, "Final_Result.xlsx")
