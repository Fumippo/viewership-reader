from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


APP_VERSION = "9.1.0-antialias-trace"
EXPECTED_READER_VERSION = "9.1.0-antialias-trace"


def load_reader_from_current_source():
    """
    Streamlitのsys.modulesキャッシュを使わず、
    App.pyと同じフォルダのreader.pyを現在の内容から直接読み込む。
    """
    reader_path = Path(__file__).resolve().with_name("reader.py")
    if not reader_path.exists():
        raise FileNotFoundError(f"reader.pyが見つかりません: {reader_path}")

    source_bytes = reader_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()[:12]
    module_name = f"viewership_reader_{source_hash}"

    if module_name in sys.modules:
        return sys.modules[module_name], source_hash, reader_path

    spec = importlib.util.spec_from_file_location(module_name, reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"reader.pyを読み込めません: {reader_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source_hash, reader_path


reader, READER_SOURCE_HASH, READER_SOURCE_PATH = load_reader_from_current_source()

st.set_page_config(
    page_title="視聴率グラフ読取",
    page_icon="📺",
    layout="wide",
)

if getattr(reader, "READER_VERSION", None) != EXPECTED_READER_VERSION:
    st.error(
        "App.pyとreader.pyの版が一致していません。"
        f"必要版: {EXPECTED_READER_VERSION} / "
        f"読込版: {getattr(reader, 'READER_VERSION', '旧版・不明')} / "
        f"読込元: {READER_SOURCE_PATH} / "
        f"ソース識別ID: {READER_SOURCE_HASH}"
    )
    st.stop()

st.title("📺 TVAL画像から1分刻み視聴率を抽出")
st.caption(
    f"アプリ: {APP_VERSION} / "
    f"解析エンジン: {reader.READER_VERSION} / "
    f"reader.py識別ID: {READER_SOURCE_HASH}"
)

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
    placeholder="例: 20:51",
)

duration = st.number_input(
    "表示時間（分）",
    min_value=30,
    max_value=360,
    value=180,
    step=1,
)

grid_percent_step = st.number_input(
    "縦軸の1目盛り（%）",
    min_value=0.5,
    max_value=20.0,
    value=2.0,
    step=0.5,
    help=(
        "縦軸が 0, 2, 4, 6… なら 2、"
        "0, 4, 8, 12… なら 4 を入力してください。"
        "ここを間違えると全局の視聴率倍率がずれます。"
    ),
)

tolerance = st.slider(
    "色検出の許容差",
    min_value=20,
    max_value=70,
    value=42,
    help=(
        "局色とのRGB差の許容範囲です。"
        "9.1では局色と白背景のアンチエイリアス混色、および他局色との競合を考慮して全幅経路追跡を行います。"
    ),
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
        st.error("時刻は20:51のように入力してください。")
        st.stop()

    end_dt = datetime.combine(end_date, parsed_time)
    raw = uploaded.getvalue()
    image = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(image)

    try:
        df, calibration, diagnostics = reader.analyze_image(
            arr=arr,
            end_time=end_dt,
            duration_minutes=int(duration),
            tolerance=int(tolerance),
            grid_percent_step=float(grid_percent_step),
        )
    except Exception as exc:
        st.error(
            "解析できませんでした。"
            "グラフ全体・0%線・少なくとも3本の水平グリッド線が"
            "画像内に入っているか確認してください。"
        )
        st.exception(exc)
        st.stop()

    st.success(
        "解析完了。9.0では各局の線を全幅で経路探索し、"
        "短い隠れやカーソル重なりを前後の線から復元します。"
    )

    diagnostic_rows = []
    for station, values in diagnostics.items():
        diagnostic_rows.append(
            {
                "局": station,
                "直接色検出率": float(values["coverage"]) * 100,
                "品質": values.get("quality", "?"),
                "最長経路推定(分)": values.get("max_gap_minutes", 0.0),
                "経路推定(px)": values.get("inferred_pixels", 0),
                "追跡方式": values.get("tracking_method", "?"),
                "アンカー": values.get("anchor", "?"),
            }
        )
    diagnostic_df = pd.DataFrame(diagnostic_rows)

    quality_values = set(diagnostic_df["品質"].astype(str))
    if "C" in quality_values:
        bad_stations = diagnostic_df.loc[
            diagnostic_df["品質"].eq("C"), "局"
        ].tolist()
        st.warning(
            "品質Cの局があります: "
            + "、".join(bad_stations)
            + "。数値は出力しますが、色証拠が薄い区間では"
            "全幅経路推定の比重が高くなっています。"
        )
    elif "B" in quality_values:
        st.info(
            "品質Bの局があります。"
            "数値は出力しますが、診断表の最長経路推定区間を確認してください。"
        )

    if calibration.time_axis_method != "hour-lines":
        st.warning(
            "正時線から時間軸を確定できなかったため、"
            "6局の色線範囲から時間軸を復元しました。"
        )

    with st.expander("自動較正・品質診断", expanded=True):
        st.json(
            {
                "解析エンジン": reader.READER_VERSION,
                "画像サイズ": f"{image.width}×{image.height}",
                "時間軸左端": round(calibration.graph_left, 1),
                "時間軸右端": round(calibration.graph_right, 1),
                "グラフ上端": round(calibration.graph_top, 1),
                "0%線": round(calibration.y_zero, 1),
                "縦軸1目盛り(%)": calibration.grid_percent_step,
                "水平グリッド間隔(px)": round(
                    calibration.grid_gap_pixels, 2
                ),
                "1%あたりpx": round(
                    calibration.pixels_per_percent,
                    3,
                ),
                "1分あたりpx": round(
                    calibration.pixels_per_minute,
                    3,
                ),
                "時間軸方式": calibration.time_axis_method,
                "採用した正時線": calibration.time_line_inliers,
            }
        )
        st.dataframe(
            diagnostic_df.style.format(
                {
                    "直接色検出率": "{:.1f}%",
                    "最長経路推定(分)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    stations = list(reader.STATION_COLORS)
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

