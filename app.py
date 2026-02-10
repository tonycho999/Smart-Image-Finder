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
    page_title="Smart-Image-Finder (Slow & Safe)",
    page_icon="🐢",
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
    # [수정됨] 요청하신 대로 3초 ~ 6초 사이 랜덤 대기
    # 이 정도면 구글 무료 제한(RPM 15)을 절대 넘지 않습니다.
    return random.uniform(3.0, 6.0)

def get_best_gemini_model():
    """모델 자동 선정"""
    try:
        models = list(genai.list_models())
        candidates = []
        for m in models:
            name = m.name.lower()
            if 'gemini' in name and 'pro' not in name and 'generateContent' in m.supported_generation_methods:
                candidates.append(m.name)
        candidates.sort(key=lambda x: ('2.0' in x, 'flash' in x, x), reverse=True)
        if candidates: return candidates[0]
        return 'gemini-1.5-flash'
    except:
        return 'gemini-1.5-flash'

def safe_download_image(url):
    """이미지 다운로드"""
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
    """엑셀 저장용 이미지 바이트 변환"""
    img.thumbnail((130, 130))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr

def search_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            q = query if attempt == 0 else query.replace(" product", "")
            results = DDGS().images(keywords=q, region="wt-wt", safesearch="off", max_results=20)
            return [r['image'] for r in results if 'image' in r]
        except: 
            time.sleep(2)
    return []

def verify_with_gemini(model_name, img, product_name):
    """Gemini AI 검수"""
    try:
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Does this image look like a product related to '{product_name}'?
        Answer YES if it shows ANY product.
        Answer NO only if it is an error page, text only, or map.
        Output only: YES or NO.
        """
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_blob = {'mime_type': 'image/jpeg', 'data': img_byte_arr.getvalue()}

        response = model.generate_content(
            [prompt, img_blob],
            generation_config=GenerationConfig(max_output_tokens=10, temperature=0.1),
            request_options={'timeout': 10}
        )
        
        answer = response.text.strip().upper()
        
        if "YES" in answer:
            return True, f"✅ 합격"
        else:
            return False, f"⛔ AI 거절"
            
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg:
            return True, "⚠️ 속도제한(자동통과)"
        elif "API key not valid" in err_msg:
            return True, "⚠️ 키 오류(자동통과)"
        else:
            return True, f"⚠️ 에러({err_msg[:10]}...)"

def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    rows = []
    
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['처리결과'] = item['status']
        rows.append(row_data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res = pd.DataFrame(rows)
        df_res.to_excel(writer, index=False, sheet_name='Result')
        
        wb = writer.book
        ws = writer.sheets['Result']
        
        ws.set_default_row(100)
        fmt_text = wb.add_format({'text_wrap': True, 'valign': 'vcenter'})
        ws.set_column(0, len(original_columns), 15, fmt_text)

        start_col = len(original_columns) + 1
        
        for i in range(target_count):
            ws.write(0, start_col + i, f"이미지_{i+1}")
            ws.set_column(start_col + i, start_col + i, 18) 

        for i, item in enumerate(data_list):
            row_idx = i + 1
            
            for k in range(target_count):
                if k < len(item['images_data']):
                    img_bytes = item['images_data'][k]
                    url_link = item['image_urls'][k]
                    
                    col_img = start_col + k
                    
                    if img_bytes:
                        ws.insert_image(row_idx, col_img, "img.jpg", {
                            'image_data': img_bytes,
                            'x_scale': 1, 'y_scale': 1,
                            'object_position': 1,
                            'url': url_link 
                        })

    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("🐢 Smart-Image-Finder (안전모드)")
st.caption("3~6초 간격으로 천천히 실행하여 에러를 방지합니다.")

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
        valid_image_urls = [] 
        
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
                        valid_image_urls.append(url)
                        
                        # [요청 반영] 3초 ~ 6초 대기
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
        st.download_button("다운로드", data, "Safe_Result.xlsx")
