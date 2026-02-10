import streamlit as st
import pandas as pd
import time
import requests
import re
import random  # [추가] 랜덤 시간 생성을 위해 필요
from io import BytesIO
from PIL import Image as PILImage
from groq import Groq
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정 & 스타일 (안전모드)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder (Pro)",
    page_icon="🛡️",
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

# ---------------------------------------------------------
# 3. 견고한 기능 함수들 (시간 설정 적용됨)
# ---------------------------------------------------------

def get_random_delay():
    """1.0초에서 3.0초 사이의 랜덤한 실수 반환 (예: 2.304초)"""
    return random.uniform(1.0, 3.0)

def safe_download_image(url):
    """이미지 다운로드 (제한시간 10초)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        # [설정] 이미지 다운로드 제한 시간 10초
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status()
        
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        
        img.thumbnail((150, 150))
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=80)
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception:
        return None

def search_with_retry(query, max_retries=3):
    """검색 실패 시 2초 -> 4초 -> 6초 대기 후 재시도"""
    for attempt in range(max_retries):
        try:
            results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=15)
            return [r['image'] for r in results]
        except Exception as e:
            # [설정] 재시도 대기 시간: 2, 4, 6초
            wait_time = 2 * (attempt + 1)
            time.sleep(wait_time)
    return []

def verify_with_retry(client, url, product_name):
    """AI 검수 (최대 10초 제한)"""
    try:
        prompt = f"Does this image clearly show the product '{product_name}'? Answer YES or NO."
        
        # [설정] AI 검수 시간 최대 10초 (timeout=10.0)
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
            temperature=0.1, 
            max_tokens=5,
            timeout=10.0 
        )
        return "YES" in completion.choices[0].message.content.upper()
    except:
        return False 

def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    rows = []
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['처리결과'] = item['status']
        rows.append(row_data)
    
    df_res = pd.DataFrame(rows)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Result')
        wb = writer.book
        ws = writer.sheets['Result']
        ws.set_default_row(100)
        
        for i, item in enumerate(data_list):
            row_idx = i + 1
            images = item['images_data']
            
            for img_idx, img_bytes in enumerate(images):
                if img_idx >= target_count: break
                col_idx = len(original_columns) + 1 + img_idx
                
                if img_bytes:
                    ws.insert_image(row_idx, col_idx, "img.jpg", {
                        'image_data': img_bytes,
                        'x_scale': 1, 'y_scale': 1,
                        'object_position': 1
                    })
                    ws.set_column(col_idx, col_idx, 20)
                    if i == 0:
                        ws.write(0, col_idx, f"이미지_{img_idx+1}")

    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("🛡️ Smart-Image-Finder (Pro)")
st.info("사람처럼 행동하는 안전 모드입니다. (랜덤 대기 시간 적용)")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])

if uploaded_file and GROQ_API_KEY:
    df = pd.read_excel(uploaded_file)
    st.write(f"총 {len(df)}개의 상품이 확인되었습니다.")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: col_brand = st.selectbox("제조사 열", df.columns, index=0)
    with c2: col_model = st.selectbox("모델명 열", df.columns, index=1 if len(df.columns)>1 else 0)
    with c3: target_count = st.number_input("필요 사진 수", 1, 5, 1)

    if st.button("🚀 작업 시작"):
        st.session_state.processed_data = [] 
        st.session_state.is_processing = True
        st.session_state.stop_requested = False
        st.rerun()

# ---------------------------------------------------------
# 5. 작업 실행 로직
# ---------------------------------------------------------
if st.session_state.is_processing:
    
    if st.button("🛑 중단하고 저장하기"):
        st.session_state.stop_requested = True
        st.warning("현재 상품까지만 처리하고 중단합니다...")

    progress_bar = st.progress(0)
    status_box = st.empty()
    
    client = Groq(api_key=GROQ_API_KEY)
    
    start_idx = len(st.session_state.processed_data)
    total_rows = len(df)
    
    for i in range(start_idx, total_rows):
        if st.session_state.stop_requested:
            break
            
        row = df.iloc[i]
        brand = str(row[col_brand])
        model = str(row[col_model])
        
        status_box.markdown(f"**[{i+1}/{total_rows}]** 처리 중: `{brand} {model}`")
        
        query = f"{brand} {model} product"
        candidates = search_with_retry(query)
        
        valid_images_bytes = []
        log_msg = ""
        
        if candidates:
            for url in candidates:
                if len(valid_images_bytes) >= target_count: break
                
                # AI 검수 (최대 10초)
                if verify_with_retry(client, url, f"{brand} {model}"):
                    img_bytes = safe_download_image(url) # 다운로드 (최대 10초)
                    if img_bytes:
                        valid_images_bytes.append(img_bytes)
                        
                        # [설정] AI 검수 및 다운로드 완료 후 랜덤 대기
                        # 1000ms ~ 3000ms 사이 (예: 2.304초)
                        human_delay = get_random_delay()
                        time.sleep(human_delay)
            
            log_msg = f"✅ {len(valid_images_bytes)}장 찾음" if valid_images_bytes else "⚠️ 검수 실패"
        else:
            log_msg = "❌ 검색 실패"
            
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images_bytes,
            'status': log_msg
        })
        
        progress_bar.progress((i + 1) / total_rows)
    
    st.session_state.is_processing = False
    st.success("작업 완료! 엑셀 파일을 다운로드하세요.")

# ---------------------------------------------------------
# 6. 다운로드 버튼
# ---------------------------------------------------------
if len(st.session_state.processed_data) > 0:
    if st.button("📥 엑셀 파일 다운로드 생성"):
        with st.spinner("엑셀 생성 중..."):
            excel_data = create_excel(
                st.session_state.processed_data, 
                df.columns.tolist(), 
                target_count
            )
            st.download_button(
                label="클릭하여 다운로드",
                data=excel_data,
                file_name="Smart_Finder_Result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
