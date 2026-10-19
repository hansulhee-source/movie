import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기 (K-Means Clustering)")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    total_count = len(df)

    # 파생 변수 생성을 위해 데이터 필터링 (결측치 및 first_week_audi == 0 제거)
    required_cols = [
        "first_scrn",
        "total_audi",
        "days_in_top10",
        "first_week_audi",
        "movieNm",
    ]
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

    # 필수 조건에 맞게 계산 가능하도록 추가 검증 (로그 취할 값들은 0 초과여야 함)
    df_clean = df_clean[
        (df_clean["first_scrn"] > 0) & (df_clean["total_audi"] > 0)
    ].copy()

    # 속성 생성 (분석에 활용)
    # 1) 스크린 수 (상용로그)
    df_clean["log_first_scrn"] = np.log10(df_clean["first_scrn"])
    # 2) 누적 관객 (상용로그)
    df_clean["log_total_audi"] = np.log10(df_clean["total_audi"])
    # 3) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20으로 클리핑)
    df_clean["long_run_index"] = (
        df_clean["total_audi"] / df_clean["first_week_audi"]
    ).clip(upper=20)
    # 4) 10위권 일수 (days_in_top10) - 그대로 유지

    filtered_count = len(df_clean)

    return df_clean, total_count, filtered_count


df, total_count, filtered_count = load_and_preprocess_data()

# 전체 편수 및 묶은 편수 표시
st.write(
    f"**전체 데이터 편수:** {total_count:,}편 | **분석 대상(묶은) 편수:** {filtered_count:,}편"
)
st.divider()

# 3. 군집화에 사용할 속성 선택
FEATURE_MAP = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

selected_feature_names = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(FEATURE_MAP.keys()),
    default=list(FEATURE_MAP.keys()),
)

if len(selected_feature_names) < 2:
    st.error("속성을 2개 이상 선택해야 분석을 진행할 수 있습니다.")
    st.stop()

selected_features = [FEATURE_MAP[name] for name in selected_feature_names]

# 4. K-Means 군집화 실행
# 선택된 속성 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# K-Means (k=3, random_state 고정)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster_temp"] = cluster_labels

# 누적 관객(total_audi) 평균을 기준으로 묶음 순서 정렬 (내림차순)
cluster_order = (
    df.groupby("cluster_temp")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index
)
label_mapping = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰",
}

df["cluster"] = df["cluster_temp"].map(label_mapping)

# 5. 2차원 산점도
st.subheader("📊 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "2차원 가로축 (X축) 속성:",
        options=list(FEATURE_MAP.keys()),
        index=0,
        key="2d_x",
    )
with col2:
    y_axis_2d = st.selectbox(
        "2차원 세로축 (Y축) 속성:",
        options=list(FEATURE_MAP.keys()),
        index=1,
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=FEATURE_MAP[x_axis_2d],
    y=FEATURE_MAP[y_axis_2d],
    color="cluster",
    hover_name="movieNm",
    labels={
        FEATURE_MAP[x_axis_2d]: x_axis_2d,
        FEATURE_MAP[y_axis_2d]: y_axis_2d,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    title=f"2차원 시각화 ({x_axis_2d} vs {y_axis_2d})",
)
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도
st.subheader("🎲 3차원 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 그리려면 위에서 속성을 3개 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "3차원 X축 속성:",
            options=selected_feature_names,
            index=0,
            key="3d_x",
        )
    with col4:
        y_axis_3d = st.selectbox(
            "3차원 Y축 속성:",
            options=selected_feature_names,
            index=1,
            key="3d_y",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "3차원 Z축 속성:",
            options=selected_feature_names,
            index=2 if len(selected_feature_names) > 2 else 0,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=FEATURE_MAP[x_axis_3d],
        y=FEATURE_MAP[y_axis_3d],
        z=FEATURE_MAP[z_axis_3d],
        color="cluster",
        hover_name="movieNm",
        labels={
            FEATURE_MAP[x_axis_3d]: x_axis_3d,
            FEATURE_MAP[y_axis_3d]: y_axis_3d,
            FEATURE_MAP[z_axis_3d]: z_axis_3d,
            "cluster": "묶음",
        },
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        title=f"3차원 시각화 ({x_axis_3d} x {y_axis_3d} x {z_axis_3d})",
    )
    # 점 크기 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 7. 묶음별 편수 및 속성 평균 표 (원래 단위)
st.subheader("📋 묶음별 통계 요약 (원래 단위)")

# 원래 단위 속성 정의
ORIGINAL_FEATURES = {
    "first_scrn": "스크린 수 평균",
    "total_audi": "누적 관객 평균",
    "days_in_top10": "10위권 일수 평균",
    "long_run_index": "롱런 지수 평균",
}

# 묶음별 계산
summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        _10위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

# 컬럼명 변경
summary_df.columns = [
    "편수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

# 숫자에 소수점 및 천단위 콤마 서식 적용
st.dataframe(
    summary_df.style.format(
        {
            "편수": "{:,}편",
            "스크린 수 평균": "{:,.1f}개",
            "누적 관객 평균": "{:,.0f}명",
            "10위권 일수 평균": "{:,.1f}일",
            "롱런 지수 평균": "{:,.2f}",
        }
    ),
    use_container_width=True,
)

# 8. 묶음별 누적 관객 수가 많은 상위 5개 영화 목록
st.subheader("🏆 묶음별 누적 관객 수 TOP 5 영화")

col_a, col_b, col_c = st.columns(3)
clusters = ["㉮", "㉯", "㉰"]
cols = [col_a, col_b, col_c]

for cluster, col in zip(clusters, cols):
    with col:
        st.markdown(f"### 묶음 {cluster}")
        top5_movies = df[df["cluster"] == cluster].nlargest(5, "total_audi")[
            ["movieNm", "total_audi"]
        ]
        top5_movies = top5_movies.reset_index(drop=True)
        top5_movies.index = top5_movies.index + 1  # 1등부터 순위 표시

        for idx, row in top5_movies.iterrows():
            st.write(f"**{idx}. {row['movieNm']}** ({row['total_audi']:,}명)")
