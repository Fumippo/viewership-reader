from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from reader import STATION_COLORS, build_calibration, build_dataframe


st.set_page_config(page_title="視聴率グラフ読取", page_icon="📺", layout="wide")
st.title("📺 TVAL画像から視聴率を1分刻みで抽出")

uploaded = st.file_uploader(
    "TVALのスクリーンショットを選択",
    type=["png", "jpg", "jpeg", "webp"],
)

end_date = st.date_input("画面下部「データ更新」の日付", value=datetime.now().date())
end_time_text = st.text_input(
    "画面下部「データ更新」の時刻",
    value=datetime.now().strftime("%H:%M"),
    placeholder="例: 19:00",
)

duration = st.number_input("表示時間（分）", 30, 360, 180, 1)
tolerance = st.slider(
    "色検出の許容差",
    8,
    50,
    24,
    help="低画質画像ほど大きめにします。通常は24前後。",
)

if uploaded is not None:
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    image = Image.open(BytesIO(raw)).convert("RGB")
    st.info(
        f"読み込んだ画像: {uploaded.name} / "
        f"{image.width}×{image.height}px / 識別ID {digest}"
    )
    st.image(image, caption="今回実際に解析する画像", use_container_width=True)

run = st.button("解析する", type="primary", use_container_width=True)

if run:
    if uploaded is None:
        st.error("先に画像を選んでください。")
        st.stop()

    try:
        parsed_time = datetime.strptime(end_time_text.strip(), "%H:%M").time()
    except ValueError:
        st.error("時刻は19:00のように入力してください。")
        st.stop()

    raw = uploaded.getvalue()
    image = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(image)
    end_dt = datetime.combine(end_date, parsed_time)

    try:
        calibration = build_calibration(
            arr,
            end_time=end_dt,
            duration_minutes=int(duration),
        )
        df, coverage = build_dataframe(
            arr,
            calibration,
            tolerance=int(tolerance),
        )
    except Exception as exc:
        st.error("解析に失敗しました。")
        st.exception(exc)
        st.stop()

    st.success("解析完了")

    low_coverage = {k: v for k, v in coverage.items() if v < 0.20}
    if low_coverage:
        st.warning(
            "線の検出率が低い局があります。結果が同じ・不自然な場合は、"
            "色検出の許容差を少し上げてください。"
        )

    with st.expander("画像別の自動較正・検出率"):
        st.json(
            {
                "画像サイズ": f"{image.width}×{image.height}",
                "グラフ左端": round(calibration.graph_left, 1),
                "グラフ右端": round(calibration.graph_right, 1),
                "グラフ上端": round(calibration.graph_top, 1),
                "0%線": round(calibration.y_zero, 1),
                "1%あたりpx": round(calibration.pixels_per_percent, 2),
                "局別検出率": {
                    k: f"{v * 100:.1f}%"
                    for k, v in coverage.items()
                },
            }
        )

    stations = list(STATION_COLORS)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["各局視聴率", "シェア率", "順位割合", "全データ"]
    )

    with tab1:
        st.line_chart(df.set_index("日時")[stations], height=480)
        st.subheader("6局合計")
        st.line_chart(df.set_index("日時")[["6局合計"]], height=300)

    with tab2:
        share_cols = [f"{s}_シェア" for s in stations]
        share_df = df.set_index("日時")[share_cols].copy()
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
        st.dataframe(rank_rates.style.format("{:.2f}%"), use_container_width=True)

    with tab4:
        st.dataframe(df, use_container_width=True, height=520)

    csv_bytes = df.to_csv(
        index=False,
        encoding="utf-8-sig",
        date_format="%Y/%m/%d %H:%M",
    ).encode("utf-8-sig")

    st.download_button(
        "CSVをダウンロード",
        csv_bytes,
        file_name=f'{end_dt.strftime("%Y%m%d_%H%M")}_viewership.csv',
        mime="text/csv",
        use_container_width=True,
    )

