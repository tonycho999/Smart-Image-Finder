import streamlit as st
import pandas as pd
import requests
import time
from io import BytesIO
from groq import Groq

# ---------------------------------------------------------
# 1. 페이지 설정 (이름: Smart-Image-Finder)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart-Image-Finder",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 사이드바 (API 키 입력)
# ---------------------------------------------------------
st.sidebar.title("⚙️ 설정 (API Keys)")
st.sidebar.markdown("작동을 위해 아래 키를 입력해주세요.")

GOOGLE_API_KEY = st.sidebar.text_input("Google API Key", type="password")
GOOGLE_CX = st.sidebar.text_input("Google Search Engine ID (CX)", type="password")
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.info("💡 키는 저장되지 않으며 새로고침 시 초기화됩니다.")

# ---------------------------------------------------------
# 3. 메인 화면 구성
# ---------------------------------------------------------
st.title("🔎 Smart-Image-Finder")
st.markdown("""
**스마트하게 엑셀 파일 속 제품의 '정확한' 사진을 찾아줍니다.**
1. **Google 검색 엔진**이 이미지를 수집하고
2. **AI(Vision)**가 제품 사진인지(흰 배경, 박스 아님 등) 검수합니다.
""")

# ---------------------------------------------------------
# 4. 기능 함수 정의
# ---------------------------------------------------------
def search_google_images(query, api_key, cx, num=3):
    """구글 Custom Search API를 통해 이미지 URL을 가져옵니다."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": cx,
        "key": api_key,
        "searchType": "image",
        "num": num,
        "safe": "active",
        "fileType": "jpg",  # JPG 선호
        "imgType": "photo"  # 사진 유형
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if "items" in data:
            return [item['link'] for item in data['items']]
        return []
    except Exception as e:
        return []

def verify_image_with_groq(image_url, product_name, api_key):
    """Groq AI(Llama 3.2 Vision)에게 이미지가 적합한지 물어봅니다."""
    try:
        client = Groq(api_key=api_key)
        
        # AI에게 보낼 프롬프트 (명령어)
        prompt = f"""
        Look at this image. Is this a clear, standalone product shot of '{product_name}'?
        
        Criteria for YES:
        1. It clearly shows the product.
        2. It has a white or plain background (preferred).
        3. It is NOT a diagram, sketch, or logo.
        4. It is NOT a box/packaging shot (unless the product is a box).
        5. It is NOT a photo of a person holding it poorly.
        
        Answer only 'YES' or 'NO'.
        """

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0,
            max_tokens=5
        )
        answer = completion.choices[0].message.content
        return "YES" in answer.upper()
    except Exception as e:
        return False

# ---------------------------------------------------------
# 5. 메인 로직 실행
# ---------------------------------------------------------
uploaded_file = st.file_uploader("엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    # 엑셀 읽기
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ 파일 업로드 성공! 데이터 미리보기:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # 컬럼 선택 UI
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_brand = st.selectbox("제조사(브랜드) 열 선택", cols, index=0 if len(cols) > 0 else 0)
    with col2:
        col_model = st.selectbox("품번(모델명) 열 선택", cols, index=1 if len(cols) > 1 else 0)

    # 실행 버튼
    if st.button("🚀 이미지 찾기 시작 (AI 검수 포함)", type="primary"):
        if not (GOOGLE_API_KEY and GOOGLE_CX and GROQ_API_KEY):
            st.error("⚠️ 왼쪽 사이드바에 API 키를 모두 입력해주세요!")
        else:
            # 진행 상황 표시줄
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_url = []
            results_status = []
            
            total = len(df)
            
            for index, row in df.iterrows():
                brand = str(row[col_brand])
                model = str(row[col_model])
                query = f"{brand} {model} product white background"
                
                status_text.markdown(f"🔍 **진행 중 ({index+1}/{total})**: `{brand} {model}` 검색...")
                
                # 1. 구글 검색 (후보 3개)
                candidates = search_google_images(query, GOOGLE_API_KEY, GOOGLE_CX, num=3)
                
                final_img = None
                verification = "유사 이미지(검수실패)"
                
                # 2. AI 검수 (순차적)
                if not candidates:
                    verification = "검색 결과 없음"
                else:
                    for img_url in candidates:
                        # Groq에게 물어보기
                        is_ok = verify_image_with_groq(img_url, f"{brand} {model}", GROQ_API_KEY)
                        if is_ok:
                            final_img = img_url
                            verification = "✅ AI 인증 완료"
                            break # 찾았으면 중단
                    
                    # AI가 다 아니라고 하면 1순위 사용 (대체)
                    if final_img is None:
                        final_img = candidates[0]
                
                results_url.append(final_img if final_img else "이미지 없음")
                results_status.append(verification)
                
                # 진행률 업데이트
                progress_bar.progress((index + 1) / total)
                time.sleep(0.2) # API 과부하 방지 딜레이

            # 결과 정리
            df["검색된_이미지_URL"] = results_url
            df["AI_검수_결과"] = results_status
            
            st.success("🎉 모든 작업이 완료되었습니다!")
            st.dataframe(df)
            
            # 다운로드 버튼 생성
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 결과 엑셀 파일 다운로드",
                data=output.getvalue(),
                file_name="Smart_Image_Finder_Result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
