import streamlit as st
import pandas as pd
import time
from io import BytesIO
from groq import Groq
from duckduckgo_search import DDGS

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder (Free)",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# 2. API 키 설정 (Groq만 필요!)
# ---------------------------------------------------------
# Streamlit Secrets에서 가져오거나, 없으면 사이드바에서 입력받음
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    st.sidebar.warning("⚠️ Groq API 키가 Secrets에 없습니다. 아래에 입력해주세요.")
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

# ---------------------------------------------------------
# 3. 메인 화면
# ---------------------------------------------------------
st.title("🔎 Smart-Image-Finder (Free Ver.)")
st.markdown("""
**구글 API 키 없이 'DuckDuckGo'를 통해 무료로 이미지를 찾습니다.**
1. **DuckDuckGo**에서 이미지를 검색하고 (무료)
2. **AI(Groq)**가 제품 사진인지 검수합니다. (현재 무료 베타)
""")

# ---------------------------------------------------------
# 4. 함수 정의
# ---------------------------------------------------------
def search_duckduckgo_images(query, num=3):
    """DuckDuckGo에서 이미지 URL을 가져옵니다. (API Key 불필요)"""
    try:
        results = DDGS().images(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            max_results=num
        )
        # 결과에서 URL만 추출
        image_urls = [r['image'] for r in results]
        return image_urls
    except Exception as e:
        print(f"DuckDuckGo 검색 에러: {e}")
        return []

def verify_image_with_groq(image_url, product_name, api_key):
    """AI가 이미지를 검수합니다."""
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
# 5. 실행 로직
# ---------------------------------------------------------
uploaded_file = st.file_uploader("엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ 파일 업로드 성공!")
        st.dataframe(df.head())
    except: st.error("파일 읽기 실패"); st.stop()

    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    with c1: col_brand = st.selectbox("제조사 열", cols, index=0)
    with c2: col_model = st.selectbox("품번 열", cols, index=1)

    if st.button("🚀 무료 이미지 검색 시작", type="primary"):
        if not GROQ_API_KEY:
            st.error("Groq API 키가 필요합니다!")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        res_url, res_status = [], []
        total = len(df)
        
        for index, row in df.iterrows():
            brand = str(row[col_brand])
            model = str(row[col_model])
            query = f"{brand} {model} product white background"
            
            status_text.text(f"({index+1}/{total}) DuckDuckGo 검색 중: {brand} {model}")
            
            # 1. 덕덕고 검색 (키 필요없음)
            candidates = search_duckduckgo_images(query, num=3)
            
            final_img, verification = None, "검수 실패"

            if candidates:
                for img in candidates:
                    # 2. AI 검수
                    if verify_image_with_groq(img, f"{brand} {model}", GROQ_API_KEY):
                        final_img, verification = img, "✅ AI 인증"
                        break
                if not final_img: final_img = candidates[0]
            else:
                verification = "검색 결과 없음"

            res_url.append(final_img or "이미지 없음")
            res_status.append(verification)
            progress_bar.progress((index + 1) / total)
            time.sleep(0.1) # 너무 빠르면 차단될 수 있으니 살짝 딜레이
            
        df["이미지URL"] = res_url
        df["검수결과"] = res_status
        
        st.success("완료!")
        st.dataframe(df)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False)
        st.download_button("📥 결과 다운로드", output.getvalue(), "Result_Free.xlsx")
