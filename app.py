import streamlit as st
import pandas as pd
import time
import requests
import re
import random
from io import BytesIO
from PIL import Image as PILImage
from groq import Groq
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder (Emergency)",
    page_icon="🚑",
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
        line-height: 1.4;
    }
    .error-msg { color: red; }
    .success-msg { color: green; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 상태 관리
# ---------------------------------------------------------
if 'processed_data' not in st.session_state: st.session_state.processed_data = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'logs' not in st.session_state: st.session_state.logs = []

def add_log(msg):
    st.session_state.logs.append(msg)

# ---------------------------------------------------------
# 3. 핵심 함수 (Llava 추가됨)
# ---------------------------------------------------------
def get_random_delay():
    return random.uniform(1.0, 3.0)

def safe_download_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((150, 150))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        return img_byte_arr
    except: return None

def search_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=10)
            return [r['image'] for r in results if 'image' in r]
        except: time.sleep(2 * (attempt + 1))
    return []

def verify_with_multi_models(client, url, product_name):
    """
    [핵심 수정] 
    1. Llama 90b (최신)
    2. Llava 7b (구형이지만 안정적)
    순서로 시도하며, 에러 메시지를 정확히 출력함.
    """
    # 11b 모델은 죽었으므로 제거함
    models_to_try = [
        "llama-3.2-90b-vision-preview", # 1순위: 최신 고성능
        "llava-v1.5-7b-4096-preview"    # 2순위: 비상용 (안정적)
    ]

    prompt = f"Does this image show '{product_name}'? Answer YES or NO."

    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
                temperature=0.1, 
                max_tokens=5,
                timeout=15.0 
            )
            return "YES" in completion.choices[0].message.content.upper()
        
        except Exception as e:
            err_msg = str(e)
            # 로그에 정확한 에러 원인 기록
            if "429" in err_msg:
                add_log(f"⚠️ {model_name}: 사용량 초과(Rate Limit). 잠시 대기...")
                time.sleep(5) # 429면 좀 오래 쉬어야 함
            elif "400" in err_msg:
                # 400 에러는 모델이 "이미지 URL을 못 읽겠다"는 뜻인 경우가 많음
                add_log(f"⚠️ {model_name}: 이미지 URL 읽기 실패 (400)")
            elif "404" in err_msg:
                add_log(f"💀 {model_name}: 모델 서비스 종료됨 (404)")
            else:
                add_log(f"⚠️ {model_name} 오류: {err_msg[:50]}...")
            
            # 다음 모델 시도
            continue

    return False # 모든 모델 실패

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
st.title("🚑 Smart-Image-Finder (Emergency Fix)")
st.caption("Llama 모델 오류 시 Llava 모델로 자동 전환하며, 상세 에러를 표시합니다.")

st.sidebar.title("상세 로그")
log_placeholder = st.sidebar.empty()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])

if uploaded_file and GROQ_API_KEY:
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
    client = Groq(api_key=GROQ_API_KEY)
    
    start_idx = len(st.session_state.processed_data)
    total_rows = len(df)
    
    for i in range(start_idx, total_rows):
        if st.session_state.stop_requested: break
            
        row = df.iloc[i]
        full_name = f"{row[col_brand]} {row[col_model]}"
        
        status_box.markdown(f"**[{i+1}/{total_rows}]** 처리 중: `{full_name}`")
        add_log(f"▶ [{i+1}] {full_name}")
        
        candidates = search_with_retry(f"{full_name} product")
        valid_images = []
        
        if candidates:
            # 최대 10개만 시도
            for url in candidates[:10]:
                if len(valid_images) >= target_count: break
                
                # [Llava 포함된 다중 검수]
                if verify_with_multi_models(client, url, full_name):
                    add_log(f"  ✅ AI 검수 통과!")
                    img_bytes = safe_download_image(url)
                    if img_bytes:
                        valid_images.append(img_bytes)
                        time.sleep(get_random_delay())
        
            msg = f"{len(valid_images)}장 확보"
            add_log(f"  🏁 결과: {msg}")
        else:
            add_log("  ❌ 검색 결과 없음 (DuckDuckGo 차단됨)")
            msg = "검색 실패"
            
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images,
            'status': msg
        })
        
        # 로그 업데이트 (최신 30줄)
        log_text = "\n".join(st.session_state.logs[-30:])
        log_placeholder.code(log_text)
        progress_bar.progress((i + 1) / total_rows)
    
    st.session_state.is_processing = False
    st.success("작업 완료!")

if len(st.session_state.processed_data) > 0:
    if st.button("📥 엑셀 파일 다운로드 생성"):
        data = create_excel(st.session_state.processed_data, df.columns.tolist(), target_count)
        st.download_button("다운로드", data, "Final_Result.xlsx")
