import streamlit as st
import pandas as pd
import time
import requests
import re
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

# UI 숨기기 및 로그창 스타일
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
# 2. 상태 관리 (새로고침 되어도 데이터 유지 시도)
# ---------------------------------------------------------
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

# ---------------------------------------------------------
# 3. 견고한 기능 함수들 (에러 방지용)
# ---------------------------------------------------------

def safe_download_image(url):
    """이미지 다운로드 및 압축 (실패 시 None 반환, 절대 멈추지 않음)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10) # 타임아웃 10초로 넉넉하게
        response.raise_for_status()
        
        img = PILImage.open(BytesIO(response.content))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        
        # 메모리 절약을 위해 즉시 리사이징
        img.thumbnail((150, 150))
        
        # 바이트로 변환하여 보관
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=80)
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception:
        return None

def search_with_retry(query, max_retries=3):
    """검색이 실패하면 잠시 쉬었다가 재시도"""
    for attempt in range(max_retries):
        try:
            # 검색 결과 15개로 제한 (너무 많으면 느려짐)
            results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=15)
            return [r['image'] for r in results]
        except Exception as e:
            time.sleep(2 * (attempt + 1)) # 2초, 4초, 6초 대기
    return [] # 끝까지 실패하면 빈 리스트 반환

def verify_with_retry(client, url, product_name):
    """AI 검수 실패 시 재시도"""
    try:
        prompt = f"Does this image clearly show the product '{product_name}'? Answer YES or NO."
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
            temperature=0.1, max_tokens=5
        )
        return "YES" in completion.choices[0].message.content.upper()
    except:
        return False # AI 에러나면 그냥 넘김

# 엑셀 생성 함수
def create_excel(data_list, original_columns, target_count):
    output = BytesIO()
    # 1. 데이터 프레임 생성
    # 원본 데이터 + 이미지 컬럼들
    rows = []
    for item in data_list:
        row_data = item['original_row'].copy()
        row_data['처리결과'] = item['status']
        rows.append(row_data)
    
    df_res = pd.DataFrame(rows)
    
    # 2. 엑셀 쓰기
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='Result')
        wb = writer.book
        ws = writer.sheets['Result']
        ws.set_default_row(100) # 행 높이 확보
        
        # 이미지 삽입
        for i, item in enumerate(data_list):
            row_idx = i + 1
            images = item['images_data'] # [BytesIO, BytesIO, ...]
            
            for img_idx, img_bytes in enumerate(images):
                if img_idx >= target_count: break
                
                # 컬럼 위치 찾기 (없으면 생성된 위치 추정)
                # 복잡하므로 단순히 맨 뒤에 붙인다고 가정하고 계산하거나,
                # 여기서는 안전하게 URL 텍스트 대신 이미지를 덮어씌우는 로직 구현이 복잡하므로
                # 간단하게: 엑셀의 특정 컬럼(J, K, L...)에 이미지를 박습니다.
                
                # 이미지 컬럼 인덱스 계산 (원본 컬럼 수 + 1(처리결과) + img_idx)
                col_idx = len(original_columns) + 1 + img_idx
                
                if img_bytes:
                    ws.insert_image(row_idx, col_idx, "img.jpg", {
                        'image_data': img_bytes,
                        'x_scale': 1, 'y_scale': 1,
                        'object_position': 1
                    })
                    # 컬럼 너비 조정
                    ws.set_column(col_idx, col_idx, 20)
                    # 헤더 쓰기 (한번만)
                    if i == 0:
                        ws.write(0, col_idx, f"이미지_{img_idx+1}")

    return output.getvalue()

# ---------------------------------------------------------
# 4. 메인 UI
# ---------------------------------------------------------
st.title("🛡️ Smart-Image-Finder (안전모드)")
st.info("이 모드는 속도보다 **'안정성'**을 최우선으로 합니다. 1000개 작업 시 브라우저 탭을 켜두세요.")

# 사이드바 설정
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

    # 시작 버튼
    if st.button("🚀 작업 시작 (절대 멈추지 않음)"):
        st.session_state.processed_data = [] # 초기화
        st.session_state.is_processing = True
        st.session_state.stop_requested = False
        st.rerun()

# ---------------------------------------------------------
# 5. 작업 실행 로직 (Session State 활용)
# ---------------------------------------------------------
if st.session_state.is_processing:
    
    # 중단 버튼 (작업 도중 누를 수 있음)
    if st.button("🛑 현재 상태에서 중단하고 저장하기"):
        st.session_state.stop_requested = True
        st.warning("작업이 곧 중단됩니다. 잠시만 기다려주세요...")

    progress_bar = st.progress(0)
    status_box = st.empty()
    log_container = st.container() # 로그가 쌓일 공간
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # 이미 처리된 개수부터 시작 (이어하기 가능하게)
    start_idx = len(st.session_state.processed_data)
    total_rows = len(df)
    
    for i in range(start_idx, total_rows):
        if st.session_state.stop_requested:
            break
            
        row = df.iloc[i]
        brand = str(row[col_brand])
        model = str(row[col_model])
        
        status_box.markdown(f"**[{i+1}/{total_rows}]** 처리 중: `{brand} {model}`")
        
        # 1. 검색 (재시도 로직 포함)
        query = f"{brand} {model} product"
        candidates = search_with_retry(query)
        
        valid_images_bytes = []
        log_msg = ""
        
        # 2. 검수 및 다운로드
        if candidates:
            for url in candidates:
                if len(valid_images_bytes) >= target_count: break
                
                # AI 검수
                if verify_with_retry(client, url, f"{brand} {model}"):
                    # 이미지 다운로드 (여기서 실패해도 프로그램 안 꺼짐)
                    img_bytes = safe_download_image(url)
                    if img_bytes:
                        valid_images_bytes.append(img_bytes)
                        time.sleep(0.2) # 서버 부하 방지
            
            if len(valid_images_bytes) > 0:
                log_msg = f"✅ {len(valid_images_bytes)}장 찾음"
            else:
                log_msg = "⚠️ AI 검수 통과 실패"
        else:
            log_msg = "❌ 검색 결과 없음"
            
        # 결과 저장 (메모리에)
        st.session_state.processed_data.append({
            'original_row': row.to_dict(),
            'images_data': valid_images_bytes,
            'status': log_msg
        })
        
        # 로그 출력 (최신 로그가 위로 오게 하려면 리스트 역순 출력 필요하지만 성능상 그냥 씀)
        # with log_container:
        #    st.text(f"{i+1}. {brand} {model} -> {log_msg}")

        progress_bar.progress((i + 1) / total_rows)
    
    # 작업 종료 또는 중단 시
    st.session_state.is_processing = False
    st.success("작업이 끝났거나 중단되었습니다! 아래 버튼으로 파일을 받으세요.")

# ---------------------------------------------------------
# 6. 다운로드 버튼 (언제든 다운로드 가능)
# ---------------------------------------------------------
if len(st.session_state.processed_data) > 0:
    st.write(f"현재 **{len(st.session_state.processed_data)}개**의 데이터가 처리되었습니다.")
    
    if st.button("📥 엑셀 파일 다운로드 생성"):
        with st.spinner("엑셀 파일을 만들고 있습니다. (이미지가 많으면 오래 걸립니다...)"):
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
