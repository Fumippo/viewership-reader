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


APP_VERSION = "7.0.2-cache-bypass"
EXPECTED_READER_VERSION = "7.0.1-hourline-layout-fix"


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

    # 同じソースなら既存モジュールを再利用し、内容が変われば別名で必ず再読込。
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
    placeholder="例: 22:44",
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
    min_value=20,
    max_value=55,
    value=42,
    help=(
        "低画質・圧縮画像に対応するため既定値を42にしています。"
        "NHK灰色だけはグリッド誤認防止の専用条件を使います。"
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
        st.error("時刻は22:44のように入力してください。")
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
        )
    except Exception as exc:
        st.error(
            "安全基準を満たさなかったため、壊れたCSVは出力しません。"
        )
        st.exception(exc)
        st.stop()

    st.success("解析完了。検出率・連続欠損の安全検査も通過しました。")

    diagnostic_rows = []
    for station, values in diagnostics.items():
        diagnostic_rows.append(
            {
                "局": station,
                "検出率": values["coverage"] * 100,
                "最長欠損(px)": values["max_gap_pixels"],
                "除外した異常ジャンプ": values["rejected_jumps"],
            }
        )
    diagnostic_df = pd.DataFrame(diagnostic_rows)

    with st.expander("自動較正・安全診断", expanded=True):
        st.json(
            {
                "解析エンジン": reader.READER_VERSION,
                "画像サイズ": f"{image.width}×{image.height}",
                "時間軸左端": round(calibration.graph_left, 1),
                "時間軸右端": round(calibration.graph_right, 1),
                "グラフ上端": round(calibration.graph_top, 1),
                "0%線": round(calibration.y_zero, 1),
                "1%あたりpx": round(
                    calibration.pixels_per_percent,
                    2,
                ),
                "1分あたりpx": round(
                    calibration.pixels_per_minute,
                    3,
                ),
            }
        )
        st.dataframe(
            diagnostic_df.style.format({"検出率": "{:.1f}%"}),
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
