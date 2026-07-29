from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from reader import STATION_COLORS, analyze_image


st.set_page_config(
    page_title="視聴率グラフ読取",
    page_icon="📺",
    layout="wide",
)

st.title("📺 TVAL画像から1分刻み視聴率を抽出")

uploaded = st.file_uploader(
    "スクリーンショットを選択",
    type=["png", "jpg", "jpeg", "webp"],
)

end_date = st.date_input(
    "画面下部「データ更新」の日付",
    value=datetime.now().date(),
)

end_time_text = st.text_input(
    "画面下部「データ更新」の時刻",
    value=datetime.now().strftime("%H:%M"),
    placeholder="例: 19:00",
)

duration = st.number_input(
    "表示時間（分）",
    min_value=30,
    max_value=360,
    value=180,
    step=1,
)

tolerance = st.slider(
    "色検出の許容差",
    min_value=8,
    max_value=50,
    value=24,
    help="元PNGは20〜24、圧縮画像は28〜36が目安です。",
)

if uploaded is not None:
    raw = uploaded.getvalue()
    image = Image.open(BytesIO(raw)).convert("RGB")
    digest = hashlib.sha256(raw).hexdigest()[:12]

    st.info(
        f"画像: {uploaded.name} / "
        f"{image.width}×{image.height}px / "
        f"識別ID: {digest}"
    )
    st.image(
        image,
        caption="今回実際に解析する画像",
        use_container_width=True,
    )

if st.button("解析する", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("画像を選択してください。")
        st.stop()

    try:
        parsed_time = datetime.strptime(
            end_time_text.strip(),
            "%H:%M",
        ).time()
    except ValueError:
        st.error("時刻は19:00のように入力してください。")
        st.stop()

    end_dt = datetime.combine(end_date, parsed_time)
    raw = uploaded.getvalue()
    image = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(image)

    try:
        df, calibration, coverage = analyze_image(
            arr=arr,
            end_time=end_dt,
            duration_minutes=int(duration),
            tolerance=int(tolerance),
        )
    except Exception as exc:
        st.error("解析に失敗しました。")
        st.exception(exc)
        st.stop()

    st.success("解析完了")

    with st.expander("自動較正・検出率"):
        st.json(
            {
                "画像サイズ": f"{image.width}×{image.height}",
                "グラフ左端": round(calibration.graph_left, 1),
                "グラフ右端": round(calibration.graph_right, 1),
                "グラフ上端": round(calibration.graph_top, 1),
                "0%線": round(calibration.y_zero, 1),
                "1%あたりpx": round(
                    calibration.pixels_per_percent,
                    2,
                ),
                "局別検出率": {
                    station: f"{rate * 100:.1f}%"
                    for station, rate in coverage.items()
                },
            }
        )

    stations = list(STATION_COLORS.keys())
    tab1, tab2, tab3, tab4 = st.tabs(
        ["各局視聴率", "シェア率", "順位割合", "全データ"]
    )

    with tab1:
        st.line_chart(
            df.set_index("日時")[stations],
            height=480,
        )
        st.subheader("6局合計")
        st.line_chart(
            df.set_index("日時")[["6局合計"]],
            height=300,
        )

    with tab2:
        share_columns = [f"{station}_シェア" for station in stations]
        share_df = df.set_index("日時")[share_columns].copy()
        share_df.columns = stations
        st.line_chart(share_df, height=480)

    with tab3:
        rank_rates = pd.DataFrame(index=stations)
        for rank in range(1, 7):
            rank_rates[f"{rank}位"] = [
                df[f"{station}_順位"].eq(rank).mean() * 100
                for station in stations
            ]

        st.bar_chart(rank_rates, height=480)
        st.dataframe(
            rank_rates.style.format("{:.2f}%"),
            use_container_width=True,
        )

    with tab4:
        st.dataframe(
            df,
            use_container_width=True,
            height=520,
        )

    csv_bytes = df.to_csv(
        index=False,
        encoding="utf-8-sig",
        date_format="%Y/%m/%d %H:%M",
    ).encode("utf-8-sig")

    st.download_button(
        "CSVをダウンロード",
        data=csv_bytes,
        file_name=f"{end_dt:%Y%m%d_%H%M}_viewership.csv",
        mime="text/csv",
        use_container_width=True,
    )

