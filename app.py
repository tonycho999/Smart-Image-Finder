import streamlit as st
import pandas as pd
import time
import re
from io import BytesIO
from groq import Groq
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정 & UI 숨기기 (깔끔하게)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder",
    page_icon="🔎",
    layout="wide"
)

# [CSS] 햄버거 메뉴, 헤더, 푸터(Manage app) 숨기기
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
    # Secrets가 없으면 사이드바에서 입력받음 (개발자용 백업)
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

# ---------------------------------------------------------
# 3. 메인 화면
# ---------------------------------------------------------
st.title("🔎 Smart-Image-Finder")

# ---------------------------------------------------------
# 4. 기능 함수 정의
# ---------------------------------------------------------
def load_google_sheet(url):
    """구글 시트 공유 링크를 엑셀 다운로드 링크로 변환하여 읽습니다."""
    # /edit 부분을 /export?format=xlsx 로 변경
    if "docs.google.com/spreadsheets" not in url:
        return None
    new_url = re.sub(r"/edit.*", "/export?format=xlsx", url)
    try:
        return pd.read_excel(new_url)
    except Exception as e:
        return None

def search_duckduckgo_images(query, num=3):
    """DuckDuckGo 이미지 검색 (무료)"""
    try:
        results = DDGS().images(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            max_results=num
        )
        return [r['image'] for r in results]
    except Exception as e:
        return []

def verify_image_with_groq(image_url, product_name, api_key):
    """AI 검수"""
    if not api_key: return False
    try:
        client = Groq(api_key=api_key)
        prompt = f"Look at this image. Is this a clear, standalone product shot of '{product_name}'? Answer YES or NO."
        
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            temperature=0, max_tokens=5
        )
        return "YES" in completion.choices[0].message.content.upper()
    except: return False

# ---------------------------------------------------------
# 5. 입력 방식 선택 (탭 기능)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📂 엑셀 파일 업로드", "🔗 구글 스프레드시트 링크"])

df = None
file_name = "Result.xlsx"

# [탭 1] 파일 업로드
with tab1:
    uploaded_file = st.file_uploader("엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx", "xls"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ 파일 로드 성공: {uploaded_file.name}")
            file_name = f"Result_{uploaded_file.name}"
        except: st.error("파일을 읽을 수 없습니다.")

# [탭 2] 링크 입력
with tab2:
    sheet_url = st.text_input("구글 스프레드시트 URL (공유 설정: '링크가 있는 모든 사용자')")
    if sheet_url:
        df = load_google_sheet(sheet_url)
        if df is not None:
            st.success("✅ 스프레드시트 로드 성공!")
            file_name = "Result_GoogleSheet.xlsx"
        else:
            st.warning("❌ 시트를 읽을 수 없습니다. URL이 올바른지, 공유 설정이 되어있는지 확인해주세요.")

# ---------------------------------------------------------
# 6. 실행 로직 (공통)
# ---------------------------------------------------------
if df is not None:
    st.markdown("---")
    st.write("### 데이터 미리보기 & 설정")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    with c1: col_brand = st.selectbox("제조사(브랜드) 열", cols, index=0 if len(cols) > 0 else 0)
    with c2: col_model = st.selectbox("품번(모델명) 열", cols, index=1 if len(cols) > 1 else 0)

    if st.button("🚀 이미지 찾기 시작", type="primary"):
        if not GROQ_API_KEY:
            st.error("⚠️ Groq API 키가 설정되지 않았습니다.")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        res_url, res_status = [], []
        total = len(df)
        
        for index, row in df.iterrows():
            brand = str(row[col_brand])
            model = str(row[col_model])
            query = f"{brand} {model} product white background"
            
            status_text.text(f"({index+1}/{total}) 검색 중: {brand} {model}")
            
            # 검색 및 검수 로직
            candidates = search_duckduckgo_images(query, num=3)
            final_img, verification = None, "검수 실패"

            if candidates:
                for img in candidates:
                    if verify_image_with_groq(img, f"{brand} {model}", GROQ_API_KEY):
                        final_img, verification = img, "✅ AI 인증"
                        break
                if not final_img: final_img = candidates[0]
            else:
                verification = "검색 결과 없음"

            res_url.append(final_img or "이미지 없음")
            res_status.append(verification)
            
            progress_bar.progress((index + 1) / total)
            time.sleep(0.1) # 딜레이
            
        df["이미지URL"] = res_url
        df["검수결과"] = res_status
        
        st.success("작업 완료!")
        st.dataframe(df)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False)
        st.download_button("📥 결과 다운로드", output.getvalue(), file_name)
