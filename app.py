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
    page_title="Smart-Image-Finder (Auto-Model)",
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
    """1000ms ~ 3000ms 사이의 랜덤한 실수 반환 (예: 2.304초)"""
    return random.uniform(1.0, 3.0)

def get_best_gemini_model():
    """
    [핵심] 사용 가능한 모델을 검색하고 Pro는 제외, 최신 Flash 우선 선택
    """
    try:
        # 모델 목록 가져오기
        models = list(genai.list_models())
        
        # 조건: 'generateContent' 지원 + 'vision' 기능(보통 gemini 시작 모델)
        # 필터: 'pro' 제외, 'gemini' 포함
        candidates = []
        for m in models:
            name = m.name.lower()
            if 'gemini' in name and 'pro' not in name and 'generateContent' in m.supported_generation_methods:
                candidates.append(m.name)
        
        # 정렬 우선순위: 숫자가 높은 것(최신) -> flash가 있는 것
        # 예: gemini-2.0-flash-exp > gemini-1.5-flash > gemini-1.5-flash-8b
        candidates.sort(key=lambda x: (
            '2.0' in x,      # 2.0 버전 우선
            'flash' in x,    # flash 우선
            x                # 이름순
        ), reverse=True)
        
        if candidates:
            return candidates[0] # 가장 좋은 것 선택
        return 'gemini-1.5-flash' # 없으면 기본값
        
    except Exception as e:
        return 'gemini-1.5-flash' # 에러나면 안전한 기본값

def safe_download_image(url):
    """이미지 다운로드 (10초 제한)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        return img
    except:
        return None

def image_to_bytes(img):
    img.thumbnail((150, 150))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

def search_with_retry(query, max_retries=3):
    """검색 실패 시 2초 -> 4초 -> 6초 대기"""
    for attempt in range(max_retries):
        try:
            results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=15)
            return [r['image'] for r in results if 'image' in r]
        except: 
            wait_time = 2 * (attempt + 1) # 2, 4, 6
            time.sleep(wait_time)
    return []

def verify_with_gemini(model_name, img, product_name):
    """AI 검수 (10초 제한 로직 포함)"""
    try:
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Does this image show the product '{product_name}'?
        Answer YES only if it clearly shows the product.
        Answer NO if it is a diagram, logo, text only, or completely different object.
        Output only one word: YES or NO.
        """
        
        # [설정] 타임아웃 10초 설정 (request_options 사용 가능 시)
        # 구글 라이브러리 버전에 따라 다르므로, 기본적으로는 모델 속도에 의존하되
        # 안전장치로 예외처리를 둠.
        response = model.generate_content(
            [prompt, img],
            generation_config=GenerationConfig(max_output_tokens=10, temperature=0.1),
            request_options={'timeout': 10} # 10초 제한
        )
        
        answer = response.text.strip().upper()
        
        if "YES" in answer:
            return True, f"✅ 합격 ({model_name})"
        else:
            return False, f"⛔ 불합격"
            
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg:
            return True, "⚠️ 속도제한 (자동통과)"
        elif "deadline" in err_msg or "timeout" in err_msg:
            return True, "⚠️ 시간초과 (자동통과)"
        return True, f"⚠️ 에러 (자동통과)"

def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    rows = []
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['처리결과'] = item['status']
        rows.append(row_data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name='Result')
        ws = writer.sheets['Result']
        ws.set_default_row(100)
        for i, item in enumerate(data_list):
            row_idx = i + 1
            for img_idx, img_bytes in enumerate(item['images_data']):
                if img_idx >= target_count: break
                col_idx = len(original_columns) + 1 + img_idx
                if img_bytes:
                    ws.insert_image(row_idx, col_idx, "img.jpg", {'image_data': img_bytes, 'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("⚡ Smart-Image-Finder (Auto-Model)")
st.caption("최적의 AI 모델을 자동으로 찾아 실행합니다. (Pro 제외, 최신 Flash 우선)")

st.sidebar.title("설정 & 로그")
use_ai_check = st.sidebar.checkbox("AI 검수 사용하기", value=True)
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
    
    # [모델 자동 선정]
    if not st.session_state.best_model_name:
        with st.spinner("최적의 모델을 검색 중입니다... (Pro 제외)"):
            st.session_state.best_model_name = get_best_gemini_model()
    
    st.info(f"🤖 현재 선택된 모델: **{st.session_state.best_model_name}**")

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
        
        if candidates:
            for url in candidates[:15]:
                if len(valid_images_bytes) >= target_count: break
                
                # 1. 이미지 다운로드 (10초 제한)
                pil_img = safe_download_image(url)
                
                if pil_img:
                    is_ok = True
                    reason = "AI 미사용"
                    
                    # 2. AI 검수 (최대 10초)
                    if use_ai_check:
                        is_ok, reason = verify_with_gemini(st.session_state.best_model_name, pil_img, full_name)
                    
                    if is_ok:
                        add_log(f"  {reason}")
                        img_bytes = image_to_bytes(pil_img)
                        valid_images_bytes.append(img_bytes)
                        
                        # [중요] 검수 완료 후 랜덤 대기 (1000ms ~ 3000ms)
                        if use_ai_check: 
                            delay = get_random_delay()
                            time.sleep(delay)
                    else:
                        add_log(f"  {reason}")
                else:
                    pass 

        msg = f"{len(valid_images_bytes)}장 확보"
        add_log(f"  🏁 결과: {msg}")
            
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images_bytes,
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
