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
# 4. 기능 함수 정의
# ---------------------------------------------------------
def load_google_sheet(url):
    if "docs.google.com/spreadsheets" not in url:
        return None
    new_url = re.sub(r"/edit.*", "/export?format=xlsx", url)
    try:
        return pd.read_excel(new_url)
    except: return None

def search_duckduckgo_images(query, num=10):
    try:
        results = DDGS().images(keywords=query, region="wt-wt", safesearch="off", max_results=num)
        return [r['image'] for r in results]
    except: return []

def verify_image_with_groq(image_url, product_name, api_key):
    if not api_key: return False
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Does this image show the product '{product_name}'?
        Answer YES only if it clearly shows the product.
        Answer NO if it is a logo, text, or completely wrong object.
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
# 5. 엑셀 생성 함수 (수정됨: 차단 방지 & 데이터 처리 강화)
# ---------------------------------------------------------
def generate_excel_with_images(df, image_cols):
    output = BytesIO()
    
    # XlsxWriter 엔진 사용
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # 행 높이 설정 (100 픽셀)
        worksheet.set_default_row(100)
        
        # 텍스트 줄바꿈 서식
        text_fmt = workbook.add_format({'text_wrap': True, 'valign': 'vcenter', 'align': 'center'})
        worksheet.set_column(0, len(df.columns) - 1, 20, text_fmt)

        # 이미지 다운로드용 헤더 (봇 차단 방지)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 데이터 순회하며 이미지 삽입
        for row_idx, row in df.iterrows():
            excel_row = row_idx + 1
            
            for col_name in image_cols:
                url = row[col_name]
                if not isinstance(url, str) or not url.startswith("http"):
                    continue
                    
                col_idx = df.columns.get_loc(col_name)
                
                try:
                    # 1. 이미지 다운로드 (타임아웃 5초)
                    response = requests.get(url, headers=headers, timeout=5)
                    response.raise_for_status()
                    
                    # 2. 이미지 데이터 가공
                    img_data = BytesIO(response.content)
                    img = PILImage.open(img_data)
                    
                    # 이미지 모드 변환 (P모드 등은 JPG 저장 시 에러 가능성 있음 -> RGB 변환)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                        
                    # 썸네일 리사이징 (메모리 절약)
                    img.thumbnail((120, 120))
                    
                    # 3. 바이트 스트림으로 다시 저장
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format="JPEG")
                    img_byte_arr.seek(0) # [중요] 포인터 초기화
                    
                    # 4. 엑셀에 삽입
                    worksheet.insert_image(excel_row, col_idx, "image.jpg", {
                        'image_data': img_byte_arr,
                        'x_scale': 1, 'y_scale': 1,
                        'object_position': 1
                    })
                except Exception as e:
                    # 실패 시 URL 텍스트만 남김 (디버깅용: print(e))
                    pass

    return output.getvalue()

# ---------------------------------------------------------
# 6. 메인 로직
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📂 엑셀 파일 업로드", "🔗 구글 스프레드시트 링크"])
df = None
file_name = "Result.xlsx"

with tab1:
    uploaded = st.file_uploader("엑셀 파일(.xlsx)", type=["xlsx", "xls"])
    if uploaded:
        try: 
            df = pd.read_excel(uploaded)
            file_name = f"Result_{uploaded.name}"
            st.success("✅ 파일 로드 성공")
        except: st.error("파일 오류")

with tab2:
    url = st.text_input("구글 스프레드시트 URL")
    if url:
        df = load_google_sheet(url)
        if df: 
            file_name = "Result_GoogleSheet.xlsx"
            st.success("✅ 시트 로드 성공")
        else: st.warning("❌ 시트 로드 실패")

if df is not None:
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 1])
    cols = df.columns.tolist()
    with c1: col_brand = st.selectbox("제조사 열", cols, index=0)
    with c2: col_model = st.selectbox("품번 열", cols, index=1 if len(cols)>1 else 0)
    with c3: target_count = st.number_input("필요한 사진 수", 1, 5, 1)

    if st.button("🚀 이미지 찾기 & 엑셀 삽입", type="primary"):
        if not GROQ_API_KEY: st.error("API 키 필요"); st.stop()

        bar = st.progress(0)
        status = st.empty()
        all_results = []
        total = len(df)
        
        # 1. 검색 단계
        for i, row in df.iterrows():
            brand = str(row[col_brand])
            model = str(row[col_model])
            status.text(f"({i+1}/{total}) 검색 중: {brand} {model}")
            
            candidates = search_duckduckgo_images(f"{brand} {model} product", num=15)
            found = []
            
            if candidates:
                for img in candidates:
                    if len(found) >= target_count: break
                    if verify_image_with_groq(img, f"{brand} {model}", GROQ_API_KEY):
                        found.append(img)
                        time.sleep(0.3)
            all_results.append(found)
            bar.progress((i+1)/total)
        
        # 2. 결과 정리
        img_cols = []
        for k in range(target_count):
            c_name = f"이미지_{k+1}"
            img_cols.append(c_name)
            df[c_name] = [res[k] if k < len(res) else "" for res in all_results]
        
        df["검수결과"] = [f"{len(r)}장 성공" for r in all_results]
        
        # 3. 엑셀 생성 단계 (시간 소요됨)
        status.text("⏳ 엑셀 파일에 이미지를 삽입하는 중입니다... (잠시만 기다려주세요)")
        try:
            excel_data = generate_excel_with_images(df, img_cols)
            st.success("🎉 완료되었습니다! 아래 버튼을 눌러 다운로드하세요.")
            st.download_button("📥 이미지 포함 엑셀 다운로드", excel_data, file_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"엑셀 생성 중 오류 발생: {e}")
