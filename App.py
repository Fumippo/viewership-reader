
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np

from reader import (
    Calibration,
    STATION_COLORS,
    build_dataframe,
    detect_horizontal_gridlines,
    estimate_y_scale,
    find_color_x_range,
    save_graphs,
)


st.set_page_config(
    page_title="視聴率グラフ読取",
    page_icon="📺",
    layout="wide",
)

st.title("📺 TVAL画像から視聴率を1分刻みで抽出")
st.caption("画像を選び、右端の日時を入れるだけです。")

uploaded = st.file_uploader(
    "TVALのスクリーンショットを選択",
    type=["png", "jpg", "jpeg", "webp"],
)

end_date = st.date_input("画像右端の日付", value=datetime.now().date())
end_time = st.time_input(
    "画像右端の時刻",
    value=datetime.now().replace(second=0, microsecond=0).time(),
    step=60,
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
    min_value=0,
    max_value=10,
    value=0,
    help="通常は0。線を検出できない場合だけ2〜5へ上げます。",
)

run = st.button("解析する", type="primary", use_container_width=True)

if run:
    if uploaded is None:
        st.error("先に画像を選んでください。")
        st.stop()

    end_dt = datetime.combine(end_date, end_time)

    try:
        image = Image.open(uploaded).convert("RGB")
        arr = np.array(image)

        color_left, color_right = find_color_x_range(arr)
        gridlines = detect_horizontal_gridlines(
            arr,
            int(color_left),
            int(color_right),
        )
        y_zero, pixels_per_percent = estimate_y_scale(gridlines)

        calibration = Calibration(
            graph_left=float(color_left),
            graph_right=float(color_right),
            y_zero=float(y_zero),
            pixels_per_percent=float(pixels_per_percent),
            end_time=end_dt,
            duration_minutes=int(duration),
        )

        with st.spinner("画像の折れ線を追跡しています…"):
            df = build_dataframe(
                arr,
                calibration,
                tolerance=int(tolerance),
            )

        stations = list(STATION_COLORS)

        st.success("解析完了")

        c1, c2, c3, c4 = st.columns(4)
        max_row = df.loc[df["6局合計"].idxmax()]
        min_row = df.loc[df["6局合計"].idxmin()]

        c1.metric(
            "合計最大",
            f'{max_row["6局合計"]:.1f}%',
            max_row["日時"].strftime("%H:%M"),
        )
        c2.metric(
            "合計最小",
            f'{min_row["6局合計"]:.1f}%',
            min_row["日時"].strftime("%H:%M"),
        )
        c3.metric(
            "開始",
            calibration.start_time.strftime("%H:%M"),
        )
        c4.metric(
            "終了",
            calibration.end_time.strftime("%H:%M"),
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["各局視聴率", "シェア率", "順位割合", "全データ"]
        )

        with tab1:
            chart_df = df.set_index("日時")[stations]
            st.line_chart(chart_df, height=480)

            total_df = df.set_index("日時")[["6局合計"]]
            st.subheader("6局合計")
            st.line_chart(total_df, height=320)

        with tab2:
            share_cols = [f"{s}_シェア" for s in stations]
            share_df = df.set_index("日時")[share_cols].copy()
            share_df.columns = stations
            st.line_chart(share_df, height=480)

            weighted_share = (
                df[stations].sum() / df["6局合計"].sum() * 100
            ).sort_values(ascending=False)

            share_table = weighted_share.rename("加重シェア率").to_frame()
            share_table["平均視聴率"] = df[stations].mean()
            share_table = share_table.sort_values(
                "加重シェア率",
                ascending=False,
            )
            st.dataframe(
                share_table.style.format(
                    {
                        "加重シェア率": "{:.2f}%",
                        "平均視聴率": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )

        with tab3:
            rank_cols = [f"{s}_順位" for s in stations]
            rank_rates = pd.DataFrame(index=stations)

            for station in stations:
                for rank in range(1, 7):
                    rank_rates[f"{rank}位"] = rank_rates.get(
                        f"{rank}位",
                        pd.Series(index=stations, dtype=float),
                    )
                    rank_rates.loc[station, f"{rank}位"] = (
                        df[f"{station}_順位"].eq(rank).mean() * 100
                    )

            st.bar_chart(rank_rates, height=480)
            st.dataframe(
                rank_rates.style.format("{:.2f}%"),
                use_container_width=True,
            )

        with tab4:
            st.dataframe(df, use_container_width=True, height=520)

        csv_bytes = df.to_csv(
            index=False,
            encoding="utf-8-sig",
            date_format="%Y/%m/%d %H:%M",
        ).encode("utf-8-sig")

        st.download_button(
            "CSVをダウンロード",
            data=csv_bytes,
            file_name=f'{end_dt.strftime("%Y%m%d_%H%M")}_viewership.csv',
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("自動較正結果"):
            st.write(
                {
                    "グラフ左端px": round(calibration.graph_left, 1),
                    "グラフ右端px": round(calibration.graph_right, 1),
                    "0%線px": round(calibration.y_zero, 1),
                    "1%あたりpx": round(
                        calibration.pixels_per_percent,
                        2,
                    ),
                }
            )

    except Exception as exc:
        st.error("解析に失敗しました。")
        st.exception(exc)
        st.info(
            "まず色検出の許容差を2〜5へ上げて再実行してください。"
        )

