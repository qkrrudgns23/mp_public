# Yeah Construction – Airport Simulation Module

공항 마스터플랜 및 시뮬레이션을 위한 Streamlit 기반 웹 애플리케이션입니다. Cirium 스케줄 데이터를 활용해 공항별 계획·분석·재배치 시뮬레이션을 수행합니다.

## 주요 기능

- **홈** – 전 세계 접근 가능 공항을 지도에서 선택하고 앱 진입
- **Masterplan** – 공항 선택, 프로파일러, 마스터플랜 작성 (피크일·시즌 등 기준 분석)
- **MNL Data Process** – MNL(다항 로짓) 데이터 전처리 및 분석
- **Airline Relocation** – 항공사 재배치 시뮬레이션
- **Apron Relocation** – 에이프론(주기장) 재배치 시뮬레이션

## 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (선택)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 앱 실행

```bash
streamlit run Home.py
```

브라우저에서 기본 주소 `http://localhost:8501` 로 접속합니다.

## 프로젝트 구조

```
├── Home.py                 # 앱 진입점 (공항 선택 지도)
├── pages/                  # Streamlit 멀티페이지
│   ├── 1_✈️_Masterplan_new.py   # 마스터플랜
│   ├── 2_📊_MNL_data_process.py # MNL 데이터 처리
│   ├── 3_🔀 Airline_Relocation.py
│   ├── 4_🔀 apron_relocation.py
│   └── 과거페이지/              # 레거시/실험 페이지
├── utils/
│   ├── masterplan.py       # 마스터플랜 로직
│   ├── cirium.py           # Cirium 데이터 연동
│   └── simulator.py        # 시뮬레이터 유틸
├── data/                   # 공항 참조, 이미지 등 정적 데이터
├── cirium/                 # Cirium 스케줄 Parquet (공항별)
├── .streamlit/
│   └── config.toml         # Streamlit 설정
└── requirements.txt
```

## 데이터

- **Cirium** – 공항 참조(`cirium_airport_ref.parquet`) 및 공항별 스케줄 Parquet (ICN, MNL, DXB 등)
- **data/** – 공항 raw 데이터, 이미지(MP_right.png, who_we_are.svg 등)

## 기술 스택

- **Streamlit** – 웹 UI
- **Plotly** – 지도·차트 시각화
- **Pandas / PyArrow** – Parquet 처리
- **Folium / streamlit-folium** – 지도
- **OpenPyXL** – 엑셀 입출력
- **SPARQLWrapper** – SPARQL 쿼리 (마스터플랜 유틸)

## 참고

- 홈에서 "Explore Your Airport→" 클릭 시 마스터플랜 페이지로 이동합니다.  
  실제 페이지 파일명이 `1_✈️_Masterplan_new.py`인 경우, `Home.py`의 `st.switch_page` 경로를 해당 파일명에 맞게 수정해야 할 수 있습니다.
