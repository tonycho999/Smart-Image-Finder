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
# 1. 페이지 설정 & 스타일
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder (Auto-Update)",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .log-box {
        height: 200px;
        overflow-y: scroll;
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 10px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 상태 관리
# ---------------------------------------------------------
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'available_models' not in st.session_state:
    st.session_state.available_models = []

def add_log(msg):
    st.session_state.logs.append(msg)

# ---------------------------------------------------------
# 3. 핵심 기능 함수들
# ---------------------------------------------------------

def get_random_delay():
    """1.0초에서 3.0초 사이의 랜덤 대기"""
    return random.uniform(1.0, 3.0)

def safe_download_image(url):
    """이미지 다운로드 (10초 제한)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status()
        
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((150, 150))
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=80)
        img_byte_arr.seek(0)
        return img_byte_arr
    except:
        return None

def search_with_retry(query, max_retries=3):
    """검색 재시도 로직"""
    for attempt in range(max_retries):
        try:
            results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=15)
            return [r['image'] for r in results if 'image' in r]
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []

# [NEW] 사용 가능한 비전 모델 자동 탐색 함수
def fetch_vision_models(client):
    """Groq API에 물어봐서 현재 사용 가능한 Vision 모델 리스트를 가져옴"""
    try:
        models = client.models.list()
        # 모델 ID에 'vision'이나 'llava'가 포함된 것만 필터링
        vision_models = [m.id for m in models.data if 'vision' in m.id or 'llava' in m.id]
        
        # 정렬 로직: '90b'가 들어간 고성능 모델을 앞으로, 나머지는 뒤로
        vision_models.sort(key=lambda x: '90b' not in x) 
        
        if not vision_models:
            # 만약 목록을 못 가져오면 기본값 강제 할당
            return ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]
            
        return vision_models
    except Exception as e:
        add_log(f"⚠️ 모델 목록 갱신 실패 (기본값 사용): {e}")
        return ["llama-3.2-90b-vision-preview"]

def verify_with_auto_model(client, url, product_name):
    """
    [핵심] 등록된 모델들을 순서대로 돌아가며 시도함.
    하나가 망가져도 다음 모델로 자동 전환.
    """
    # 세션에 저장된 모델 리스트가 없으면 가져옴
    if not st.session_state.available_models:
        st.session_state.available_models = fetch_vision_models(client)
        add_log(f"📋 사용 가능 모델: {st.session_state.available_models}")

    prompt = f"""
    Does this image show the product '{product_name}'?
    If it looks even slightly like the product, answer YES.
    Answer NO only if it is completely wrong.
    Answer YES or NO.
    """

    # 모델 리스트를 순회하며 시도
    for model_name in st.session_state.available_models:
        try:
            completion = client.chat.completions.create(
                model=model_name, # 여기서 모델을 갈아끼움
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
                temperature=0.1, 
                max_tokens=5,
                timeout=10.0 
            )
            return "YES" in completion.choices[0].message.content.upper()
        
        except Exception as e:
            # 에러 발생 시 로그 남기고 다음 모델로 넘어감
            error_msg = str(e)
            if "model_decommissioned" in error_msg or "404" in error_msg or "400" in error_msg:
                add_log(f"⚠️ 모델({model_name}) 실패 -> 다음 모델 시도 중...")
                continue # 다음 모델로!
            else:
                return False # 모델 문제가 아닌 다른 에러면 그냥 실패 처리

    return False # 모든 모델이 실패하면 False

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
                    ws.insert_image(row_idx, col_idx, "img.jpg", {
                        'image_data': img_bytes, 'x_scale': 1, 'y_scale': 1, 'object_position': 1
                    })
                    if i == 0: ws.write(0, col_idx, f"이미지_{img_idx+1}")
    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("🤖 Smart-Image-Finder (Auto-Update)")
st.caption("새로운 AI 모델이 나오면 자동으로 찾아내어 적용합니다.")

# 사이드바 (로그창)
st.sidebar.title("작업 로그")
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
        st.session_state.available_models = [] # 모델 리스트 초기화 (새로 검색)
        st.rerun()

# ---------------------------------------------------------
# 5. 작업 실행 로직
# ---------------------------------------------------------
if st.session_state.is_processing:
    
    if st.button("🛑 중단하고 저장하기"):
        st.session_state.stop_requested = True
        st.warning("마무리 중입니다...")

    progress_bar = st.progress(0)
    status_box = st.empty()
    client = Groq(api_key=GROQ_API_KEY)
    
    # [시작 시] 모델 목록 자동 갱신
    if not st.session_state.available_models:
        with st.spinner("최신 AI 모델 목록을 받아오는 중..."):
            st.session_state.available_models = fetch_vision_models(client)
            add_log(f"✅ 모델 자동 감지 완료: {len(st.session_state.available_models)}개 발견")

    start_idx = len(st.session_state.processed_data)
    total_rows = len(df)
    
    for i in range(start_idx, total_rows):
        if st.session_state.stop_requested: break
            
        row = df.iloc[i]
        full_name = f"{row[col_brand]} {row[col_model]}"
        
        status_box.markdown(f"**[{i+1}/{total_rows}]** 처리 중: `{full_name}`")
        add_log(f"[{i+1}] {full_name} 검색")
        
        query = f"{full_name} product"
        candidates = search_with_retry(query)
        
        valid_images_bytes = []
        log_msg = ""
        
        if candidates:
            for url in candidates[:12]: # 최대 12개 검토
                if len(valid_images_bytes) >= target_count: break
                
                # [여기가 핵심] 자동으로 모델 돌려가며 검수
                if verify_with_auto_model(client, url, full_name):
                    img_bytes = safe_download_image(url)
                    if img_bytes:
                        valid_images_bytes.append(img_bytes)
                        time.sleep(get_random_delay())
            
            log_msg = f"{len(valid_images_bytes)}장 찾음"
            add_log(f" -> {log_msg}")
        else:
            add_log(" -> 검색 결과 없음")
            log_msg = "검색 실패"
            
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images_bytes,
            'status': log_msg
        })
        
        log_text = "\n".join(st.session_state.logs[-20:])
        log_placeholder.code(log_text)
        progress_bar.progress((i + 1) / total_rows)
    
    st.session_state.is_processing = False
    st.success("작업 완료!")

if len(st.session_state.processed_data) > 0:
    if st.button("📥 엑셀 파일 다운로드 생성"):
        with st.spinner("엑셀 생성 중..."):
            data = create_excel(st.session_state.processed_data, df.columns.tolist(), target_count)
            st.download_button("다운로드", data, "Auto_Update_Result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
