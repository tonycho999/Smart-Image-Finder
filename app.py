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
st.set_page_config(page_title="Smart-Image-Finder (Debug)", page_icon="🛠️", layout="wide")

# 로그 스타일
st.markdown("""
<style>
    .log-text {font-family: monospace; font-size: 12px; color: #333;}
    .success {color: green; font-weight: bold;}
    .fail {color: red; font-weight: bold;}
    .info {color: blue;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 상태 관리
# ---------------------------------------------------------
if 'logs' not in st.session_state: st.session_state.logs = []
if 'processed_data' not in st.session_state: st.session_state.processed_data = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = False

def add_log(msg):
    st.session_state.logs.append(msg)

# ---------------------------------------------------------
# 3. 핵심 함수 (로그 추가됨)
# ---------------------------------------------------------
def get_random_delay():
    return random.uniform(1.0, 3.0)

def search_with_retry(query):
    """검색 실패 원인 파악"""
    try:
        # max_results를 10개로 설정
        results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=10)
        urls = [r['image'] for r in results if 'image' in r]
        return urls
    except Exception as e:
        add_log(f"❌ 검색 에러 발생: {str(e)}")
        return []

def verify_with_retry(client, url, product_name):
    """AI 검수 (기준 완화 & 로그 출력)"""
    try:
        # 프롬프트를 아주 단순하게 변경 (일단 YES를 유도)
        prompt = f"""
        Is this an image of '{product_name}'? 
        If it looks even slightly like the product, answer YES.
        Only answer NO if it is completely wrong (like a cat, a car, or random text).
        Answer YES or NO.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
            temperature=0.1, max_tokens=5, timeout=10.0
        )
        
        answer = completion.choices[0].message.content.upper()
        # 로그에 AI가 뭐라 했는지 기록
        if "YES" in answer:
            return True, answer
        else:
            return False, answer # NO라고 답함
    except Exception as e:
        return False, f"Error: {str(e)}"

def safe_download_image(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        img.thumbnail((150, 150))
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        return img_byte_arr
    except: return None

# 엑셀 생성 (생략없이 복원)
def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    rows = []
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['결과메시지'] = item['status']
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
# 4. UI 구성
# ---------------------------------------------------------
st.title("🛠️ Smart-Image-Finder (진단 모드)")
st.info("오른쪽 사이드바의 로그를 확인하세요.")

# 사이드바 (로그창)
st.sidebar.title("📋 실시간 로그")
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

    if st.button("🚀 진단 시작"):
        st.session_state.logs = []
        st.session_state.processed_data = []
        st.session_state.is_processing = True
        st.rerun()

# ---------------------------------------------------------
# 5. 실행 로직
# ---------------------------------------------------------
if st.session_state.is_processing:
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    client = Groq(api_key=GROQ_API_KEY)
    
    start_idx = len(st.session_state.processed_data)
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        brand = str(row[col_brand])
        model = str(row[col_model])
        full_name = f"{brand} {model}"
        
        status_text.text(f"처리 중: {full_name}")
        add_log(f"--- [ {full_name} ] 시작 ---")
        
        # 1. 검색
        query = f"{full_name} product"
        candidates = search_with_retry(query)
        add_log(f"🔍 검색결과: {len(candidates)}개 발견")
        
        valid_images = []
        
        # 2. 검수
        if candidates:
            for idx, url in enumerate(candidates):
                if len(valid_images) >= target_count: break
                
                # 로그에 이미지 URL 일부 출력
                short_url = url[:30] + "..."
                add_log(f"  [{idx+1}] 검수 시도: {short_url}")
                
                is_valid, ai_reason = verify_with_retry(client, url, full_name)
                
                if is_valid:
                    add_log(f"  ✅ AI 합격! (응답: {ai_reason})")
                    img_bytes = safe_download_image(url)
                    if img_bytes:
                        valid_images.append(img_bytes)
                        add_log("  📥 다운로드 성공")
                        time.sleep(get_random_delay())
                    else:
                        add_log("  ❌ 다운로드 실패 (접근 거부됨)")
                else:
                    add_log(f"  ⛔ AI 불합격 (응답: {ai_reason})")
                    # 실패해도 너무 빠르면 차단되니 살짝 대기
                    time.sleep(0.5)
        else:
            add_log("❌ 검색 결과가 아예 없습니다. (DuckDuckGo 차단 의심)")

        # 결과 저장
        status_msg = f"{len(valid_images)}장 성공"
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images,
            'status': status_msg
        })
        
        # 사이드바 로그 업데이트 (최신 20줄만 보여주기)
        log_text = "\n".join(st.session_state.logs[-30:])
        log_placeholder.code(log_text)
        
        progress_bar.progress((i + 1) / len(df))
    
    st.session_state.is_processing = False
    st.success("진단 완료!")

# 다운로드
if len(st.session_state.processed_data) > 0:
    if st.button("📥 결과 파일 다운로드"):
        data = create_excel(st.session_state.processed_data, df.columns.tolist(), target_count)
        st.download_button("Download", data, "Debug_Result.xlsx")
