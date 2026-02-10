import streamlit as st
import pandas as pd
import time
import requests
import re
import random
from io import BytesIO
from PIL import Image as PILImage
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Smart-Image-Finder (Debug)", page_icon="🔧", layout="wide")

# ---------------------------------------------------------
# 2. API 키 불러오기 (디버깅 추가)
# ---------------------------------------------------------
try:
    HF_API_KEY = st.secrets["HF_API_KEY"]
    st.sidebar.success("✅ Secrets에서 키를 찾았습니다!")
except Exception as e:
    st.sidebar.warning("⚠️ Secrets에서 키를 못 찾았습니다. 아래에 입력해주세요.")
    st.sidebar.error(f"에러 내용: {e}") # 여기서 왜 못 읽었는지 알려줌
    HF_API_KEY = st.sidebar.text_input("Hugging Face Token (hf_...)", type="password")

# ---------------------------------------------------------
# 3. 핵심 함수
# ---------------------------------------------------------
def get_random_delay():
    return random.uniform(2.0, 3.0)

def safe_download_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        return img
    except: return None

def image_to_bytes(img):
    img.thumbnail((130, 130))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

def search_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            q = query if attempt == 0 else query.replace(" product", "")
            results = DDGS().images(keywords=q, region="wt-wt", safesearch="off", max_results=15)
            return [r['image'] for r in results if 'image' in r]
        except: time.sleep(2)
    return []

def verify_with_huggingface(api_key, img_bytes, brand_name):
    # BLIP 모델 사용 (이미지 설명)
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(API_URL, headers=headers, data=img_bytes, timeout=10)
        
        # [에러 진단]
        if response.status_code == 503:
            return True, "⚠️ 모델 로딩중(503/자동통과)" # 무료라서 모델 켜지는 중
        elif response.status_code == 401:
            return True, "⚠️ 키 오류(401/자동통과)" # 키가 틀림
        elif response.status_code != 200:
            return True, f"⚠️ API에러({response.status_code})"

        result = response.json()
        
        if isinstance(result, list) and 'generated_text' in result[0]:
            caption = result[0]['generated_text'].lower()
            if brand_name.lower().split()[0] in caption or "shoes" in caption or "product" in caption:
                 return True, f"✅ 합격"
            else:
                 return True, f"⚠️ 애매함({caption[:10]}..)"
        
        return True, "⚠️ 분석불가"

    except Exception as e:
        return True, f"⚠️ 시스템에러({str(e)})"

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
        start_col = len(original_columns) + 1
        
        for i, item in enumerate(data_list):
            row_idx = i + 1
            for k in range(target_count):
                if k < len(item['images_data']):
                    img_bytes = item['images_data'][k]
                    url_link = item['image_urls'][k]
                    col_img = start_col + k
                    if img_bytes:
                        ws.insert_image(row_idx, col_img, "img.jpg", {
                            'image_data': img_bytes, 'x_scale': 1, 'y_scale': 1, 'object_position': 1,
                            'url': url_link 
                        })
    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("🔧 Smart-Image-Finder (Debug Mode)")

if 'processed_data' not in st.session_state: st.session_state.processed_data = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = False

uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])

if uploaded_file and HF_API_KEY:
    df = pd.read_excel(uploaded_file)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: col_brand = st.selectbox("제조사 열", df.columns, index=0)
    with c2: col_model = st.selectbox("모델명 열", df.columns, index=1 if len(df.columns)>1 else 0)
    with c3: target_count = st.number_input("필요 사진 수", 1, 5, 1)

    if st.button("🚀 작업 시작"):
        st.session_state.processed_data = [] 
        st.session_state.is_processing = True
        st.rerun()

if st.session_state.is_processing:
    progress_bar = st.progress(0)
    status_box = st.empty()
    
    for i, row in df.iterrows():
        brand = str(row[col_brand])
        model = str(row[col_model])
        status_box.text(f"처리 중: {brand} {model}")
        
        candidates = search_with_retry(f"{brand} {model} product")
        valid_bytes = []
        valid_urls = []
        
        if candidates:
            for url in candidates[:15]:
                if len(valid_bytes) >= target_count: break
                pil_img = safe_download_image(url)
                if pil_img:
                    # HuggingFace 전송용 변환
                    buf = BytesIO()
                    pil_img.save(buf, format='JPEG')
                    
                    is_ok, reason = verify_with_huggingface(HF_API_KEY, buf.getvalue(), brand)
                    
                    # 에러나도 저장 (진단용)
                    final_bytes = image_to_bytes(pil_img)
                    valid_bytes.append(final_bytes)
                    valid_urls.append(url)
                    time.sleep(get_random_delay())

        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_bytes,
            'image_urls': valid_urls,
            'status': f"{len(valid_bytes)}장"
        })
        progress_bar.progress((i + 1) / len(df))
    
    st.session_state.is_processing = False
    st.success("완료!")

if len(st.session_state.processed_data) > 0:
    data = create_excel(st.session_state.processed_data, df.columns.tolist(), target_count)
    st.download_button("다운로드", data, "Debug_Result.xlsx")
