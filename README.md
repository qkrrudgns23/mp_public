# Airport airside layout & simulation

레이아웃 디자이너·저장 API·시뮬레이션 연동을 **표준 라이브러리 HTTP 서버**로 제공합니다.

## 주요 기능

- **홈** (`/home`) – BluPrint 글로브 (공항 참조 Parquet 기반)
- **Layout Design** (`/`) – 터미널·에어사이드 디자이너
- **REST API** – 레이아웃 저장/로드, 지도/OSM 연동, Pro Sim 등 (`utils/layout_receiver.py` 참고)

## 실행 방법

### 1. 환경 설정

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. 앱 실행 (UI + 레이아웃 API 단일 포트)

```bash
python run_app.py
```

브라우저에서 `http://127.0.0.1:8501/` (디자이너), `http://127.0.0.1:8501/home` (글로브) 로 접속합니다.  
외부 접속(EC2 등)은 `HOST=0.0.0.0 PORT=8501 python run_app.py` 후 보안 그룹에서 해당 포트를 허용하세요.

**레이아웃 서버만** (기본 8765): `python run_layout_server.py`

## 프로젝트 구조 (요약)

```
├── run_app.py              # 단일 포트 웹/UI + 레이아웃 API (권장 엔트리)
├── run_layout_server.py    # 디자이너·API만 (포트 기본 8765)
├── pages/Layout_Design/    # 디자이너 정적 에셋·스크립트
├── utils/
│   ├── layout_receiver.py  # HTTP 핸들러 및 API 구현
│   ├── layout_design_build.py / layout_designer_standalone.py
│   └── airside_sim.py      # 시뮬레이션 코어
├── data/
└── requirements.txt
```

## 데이터

- 공항 참조 Parquet 및 레이아웃·결과 저장은 ``data/`` 아래 경로 규약을 따릅니다.

## 기술 스택

- **Python HTTPServer** – 웹·API 제공
- **Pandas / PyArrow** – Parquet 처리
- **three.js 등 (CDN)** – 홈 글로브
