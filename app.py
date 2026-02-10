import streamlit as st
import pandas as pd
import time
import re
import requests
from io import BytesIO
from PIL import Image as PILImage
from groq import Groq
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정 & UI 숨기기
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder",
    page_icon="🔎",
    layout="wide"
)

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. API 키 설정 (Groq)
# ---------------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

# ---------------------------------------------------------
# 3. 메인 화면
# ---------------------------------------------------------
st.title("🔎 Smart-Image-Finder")

# ---------------------------------------------------------
# 4. 기능 함수 정의 (검색 & AI)
# ---------------------------------------------------------
def load_google_sheet(url):
    """구글 시트 읽기"""
    if "docs.google.com/spreadsheets" not in url:
        return None
    new_url = re.sub(r"/edit.*", "/export?format=xlsx", url)
    try:
        return pd.read_excel(new_url)
    except: return None

def search_duckduckgo_images(query, num=10):
    """이미지 검색 (DuckDuckGo)"""
    try:
        results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=num)
        return [r['image'] for r in results]
    except: return []

def verify_image_with_groq(image_url, product_name, api_key):
    """AI 이미지 검수"""
    if not api_key: return False
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Does this image show the product '{product_name}'?
        Answer YES only if it clearly shows the product itself.
        Answer NO if it is a logo, text, diagram, or completely wrong object.
        Answer only YES or NO.
        """
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}],
            temperature=0.1, max_tokens=5
        )
        return "YES" in completion.choices[0].message.content.upper()
    except: return False

# ---------------------------------------------------------
# 5. 엑셀 생성 함수 (이미지 삽입 기능 포함) ⭐ 중요
# ---------------------------------------------------------
def generate_excel_with_images(df, image_cols):
    """데이터프레임을 받아 이미지를 실제 셀에 삽입하여 엑셀 바이너리를 반환"""
    output = BytesIO()
    
    # Pandas ExcelWriter를 xlsxwriter 엔진으로 생성
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # 서식 설정 (텍스트 줄바꿈, 수직 가운데 정렬)
        text_format = workbook.add_format({'text_wrap': True, 'valign': 'vcenter'})
        
        # 전체 행 높이 설정 (이미지가 들어갈 공간 확보, 약 100픽셀)
        worksheet.set_default_row(80) 
        
        # 전체 열에 서식 적용 (A열부터 끝까지)
        worksheet.set_column(0, len(df.columns) - 1, 20, text_format)

        # 이미지 컬럼들 처리
        # 데이터프레임의 컬럼 이름을 보고 엑셀의 몇 번째 열인지 찾음
        col_indices = [df.columns.get_loc(c) for c in image_cols]

        for row_idx, row in df.iterrows():
            # 엑셀은 헤더가 0행이므로 데이터는 1행부터 시작
            excel_row = row_idx + 1
            
            for col_name in image_cols:
                url = row[col_name]
                col_idx = df.columns.get_loc(col_name)
                
                # URL이 있고 "검색실패"가 아니면 이미지 다운로드 시도
                if url and str(url).startswith("http"):
                    try:
                        response = requests.get(url, timeout=3)
                        if response.status_code == 200:
                            img_data = BytesIO(response.content)
                            
                            # Pillow로 이미지 리사이징 (용량 최적화 & 셀 맞춤)
                            img = PILImage.open(img_data)
                            img.thumbnail((120, 120)) # 썸네일 크기
                            
                            # 메모리에 저장된 이미지를 다시 바이트로 변환
                            img_byte_arr = BytesIO()
                            img_format = img.format if img.format else 'JPEG'
                            img.save(img_byte_arr, format=img_format)
                            
                            # 엑셀에 삽입
                            worksheet.insert_image(excel_row, col_idx, url, {
                                'image_data': img_byte_arr,
                                'x_scale': 1, 'y_scale': 1,
                                'object_position': 1 # 셀 내 이동/크기변함 설정
                            })
                            
                            # 이미지 들어간 열 너비 조금 넓게
                            worksheet.set_column(col_idx, col_idx, 18)
                    except:
                        pass # 이미지 다운로드 실패 시 그냥 URL 텍스트만 유지

    return output.getvalue()

# ---------------------------------------------------------
# 6. 입력 UI
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📂 엑셀 파일 업로드", "🔗 구글 스프레드시트 링크"])

df = None
file_name = "Result.xlsx"

with tab1:
    uploaded_file = st.file_uploader("엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx", "xls"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ 파일 로드: {uploaded_file.name}")
            file_name = f"Result_{uploaded_file.name}"
        except: st.error("파일 오류")

with tab2:
    sheet_url = st.text_input("구글 스프레드시트 URL")
    st.caption("결과물은 엑셀 파일로 다운로드됩니다. (이미지 포함)")
    if sheet_url:
        df = load_google_sheet(sheet_url)
        if df is not None:
            st.success("✅ 시트 로드 성공")
            file_name = "Result_GoogleSheet.xlsx"
        else: st.warning("❌ 시트 로드 실패")

# ---------------------------------------------------------
# 7. 실행 로직
# ---------------------------------------------------------
if df is not None:
    st.markdown("---")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    cols = df.columns.tolist()
    
    with c1: col_brand = st.selectbox("제조사 열", cols, index=0)
    with c2: col_model = st.selectbox("품번 열", cols, index=1 if len(cols) > 1 else 0)
    with c3: 
        target_count = st.number_input("필요한 사진 수", min_value=1, max_value=5, value=1)

    if st.button("🚀 이미지 찾기 시작 (이미지 엑셀 삽입)", type="primary"):
        if not GROQ_API_KEY:
            st.error("⚠️ Groq API 키 필요"); st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        total = len(df)
        
        # --- 검색 루프 ---
        for index, row in df.iterrows():
            brand = str(row[col_brand])
            model = str(row[col_model])
            query = f"{brand} {model} product"
            
            status_text.text(f"({index+1}/{total}) 검색 및 다운로드 준비 중: {brand} {model}")
            
            candidates = search_duckduckgo_images(query, num=15)
            found_images = []
            
            if candidates:
                for img in candidates:
                    if len(found_images) >= target_count: break
                    if verify_image_with_groq(img, f"{brand} {model}", GROQ_API_KEY):
                        found_images.append(img)
                        time.sleep(0.3) 
            
            all_results.append(found_images)
            progress_bar.progress((index + 1) / total)
            
        # --- 결과 정리 ---
        image_columns = []
        for i in range(target_count):
            col_name = f"이미지_{i+1}"
            image_columns.append(col_name)
            df[col_name] = [res[i] if i < len(res) else "" for res in all_results]

        df["검수_상태"] = [f"{len(res)}장 찾음" for res in all_results]
        
        st.success("🎉 검색 완료! 엑셀 파일 생성 중... (이미지 삽입에 시간이 조금 걸립니다)")
        
        # --- 엑셀 생성 (이미지 삽입) ---
        excel_data = generate_excel_with_images(df, image_columns)
        
        st.download_button(
            label="📥 이미지가 포함된 엑셀 다운로드",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
