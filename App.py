"""TVAL now 画像 → 1分刻み視聴率（Streamlit UI）。

使い方:
    streamlit run App.py

9.2.1 reader.py と連携する表示用アプリ。CSVは旧版と同じ27列。
画像プレビューはアップロード直後に表示。CSVダウンロード後は
アップロード済み画像・解析画面を自動クリアして次の画像を選べる。
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

APP_VERSION = "9.2.1-adaptive-range-ui-fix"
EXPECTED_READER_VERSION = "9.2.0-optimized-trace"


def load_reader_from_current_source():
    """App.py と同じ場所の reader.py を内容ハッシュ付きで読み込む。"""
    path = Path(__file__).resolve().with_name("reader.py")
    if not path.exists():
        raise FileNotFoundError(f"reader.py が見つかりません: {path}")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    module_name = f"viewership_reader_{source_hash}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"reader.py を読み込めません: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[module_name], source_hash, path


reader, reader_hash, reader_path = load_reader_from_current_source()
st.set_page_config(page_title="TVAL画像視聴率読取", page_icon="📺", layout="wide")
if (
    getattr(reader, "READER_VERSION", None) != EXPECTED_READER_VERSION
    or not hasattr(reader, "READER_PATCH_VERSION")
):
    st.error(
        "改良版のreader.pyが必要です。"
        f" 読込版={getattr(reader, 'READER_VERSION', '不明')}"
        f" / patch={getattr(reader, 'READER_PATCH_VERSION', '不明')}"
    )
    st.stop()

# Streamlitのfile_uploaderは「値をNoneに書き戻す」ことで消去できない。
# 代わりにキーを更新し、新しい空のアップローダーを自動生成する。
if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def on_image_changed():
    """新しい画像を選んだら、旧画像の解析結果を表示しない。"""
    st.session_state.analysis_result = None


def on_main_csv_download():
    """CSV保存ボタンを押したら、次の画像のために画面を自動リセットする。"""
    st.session_state.upload_generation += 1
    st.session_state.analysis_result = None


def parse_end_values(text: str):
    if not text.strip():
        return None
    parts = [
        p.strip().replace("％", "").replace("%", "")
        for p in text.replace("、", ",").split(",")
    ]
    if len(parts) != 6:
        raise ValueError("画面表示値はNHK、日テレ、テレ朝、TBS、テレ東、フジの順に6個入力してください。")
    return [float(p) for p in parts]


def trace_overlay(image, traces, calib):
    """抽出した曲線を元画像に重ねて見せる（必要時のみ計算）。"""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    palette = {
        "NHK総合": (70, 70, 70),
        "日本テレビ": (235, 166, 0),
        "テレビ朝日": (210, 48, 190),
        "TBS": (0, 88, 240),
        "テレビ東京": (20, 35, 108),
        "フジテレビ": (215, 35, 35),
    }
    for station, (xs, ys, observed) in traces.items():
        points = [(int(x), int(round(float(y)))) for x, y in zip(xs, ys)]
        if len(points) >= 2:
            draw.line(points, fill=palette[station], width=2)
        invalid = np.flatnonzero(~observed)
        if len(invalid):
            groups = np.split(invalid, np.where(np.diff(invalid) != 1)[0] + 1)
            for group in groups:
                if len(group) >= max(8, int(calib.pixels_per_minute * 2)):
                    mid = int(group[len(group) // 2])
                    x, y = int(xs[mid]), int(round(float(ys[mid])))
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black")
    return out


st.title("📺 TVAL now 画像 → 1分刻み視聴率")
st.caption(
    f"UI {APP_VERSION} / reader {reader.READER_VERSION} / "
    f"改良版 {reader.READER_PATCH_VERSION} / ソース識別ID {reader_hash}"
)

# ------------------------ 入力 ------------------------
col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded = st.file_uploader(
        "スクリーンショットを選択",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"tval_image_{st.session_state.upload_generation}",
        on_change=on_image_changed,
    )
    end_date = st.date_input("『データ更新』の日付", value=None)
    end_time_text = st.text_input("『データ更新』の時刻", value="", placeholder="例: 21:15")
    duration = st.number_input("グラフ表示時間（分）", min_value=30, max_value=360, value=180)
    grid_step = st.number_input(
        "縦軸1目盛り（%）", min_value=0.5, max_value=20.0, value=2.0, step=0.5,
        help="0,2,4,…なら2。0,4,8,…なら4。画像の縦軸に合わせて指定します。",
    )
with col_right:
    mode = st.selectbox(
        "色線解析モード", ["auto", "strict", "high-range"],
        format_func=lambda value: {
            "auto": "自動（推奨）",
            "strict": "従来の厳格24",
            "high-range": "高レンジ35（検証用）",
        }[value],
    )
    rgb_tolerance = st.slider("カーソルRGB差", 10, 50, 20)
    # デバッグ用の画像重ね合わせは必要時だけ表示。通常作業の操作は増やさない。
    show_overlay = st.checkbox("解析線を元画像に重ねて検証", value=False)
    known_endpoint = st.text_input(
        "画面上部の6局の数値（任意）",
        placeholder="例: 3.2, 5.7, 6.5, 20.0, 3.8, 3.2",
        help="NHK、日テレ、テレ朝、TBS、テレ東、フジの順です。",
    )

raw = uploaded.getvalue() if uploaded is not None else None

if st.button("▶ 解析する", type="primary", use_container_width=True):
    if raw is None or end_date is None:
        st.error("画像と『データ更新』の日付を指定してください。")
    else:
        try:
            parsed_time = datetime.strptime(end_time_text.strip(), "%H:%M").time()
            known = parse_end_values(known_endpoint)
        except ValueError as exc:
            st.error(str(exc))
        else:
            end_dt = datetime.combine(end_date, parsed_time)
            image = Image.open(BytesIO(raw)).convert("RGB")
            try:
                with st.spinner("視聴率グラフを解析しています…"):
                    df, calib, diagnostics, traces = reader.analyze_image(
                        arr=np.asarray(image),
                        end_time=end_dt,
                        duration_minutes=int(duration),
                        tolerance=int(rgb_tolerance),
                        grid_percent_step=float(grid_step),
                        evidence_mode=mode,
                        return_traces=True,
                    )
            except Exception as exc:
                st.session_state.analysis_result = None
                st.error("解析に失敗しました。グラフ全体・0%線・水平グリッド線を確認してください。")
                st.exception(exc)
            else:
                # ファイル保存などによるStreamlitの再実行でも結果を維持する。
                # ただしCSV保存後はon_main_csv_downloadで自動消去する。
                st.session_state.analysis_result = {
                    "df": df,
                    "calib": calib,
                    "diagnostics": diagnostics,
                    "traces": traces,
                    "end_dt": end_dt,
                    "known": known,
                    "image_bytes": raw,
                }

# プレビューは折り畳まず自動表示。ただし解析ボタンより下・小さめに置き、
# スクロールしないと「解析する」を押せないという逆効果を避ける。
if raw is not None:
    preview = Image.open(BytesIO(raw)).convert("RGB")
    st.caption(f"画像: {uploaded.name} / {preview.width}×{preview.height}px")
    st.image(preview, width=min(650, preview.width))

# ------------------------ 結果 ------------------------
result = st.session_state.analysis_result
if result is not None:
    df = result["df"]
    calib = result["calib"]
    diagnostics = result["diagnostics"]
    traces = result["traces"]
    end_dt = result["end_dt"]
    known = result["known"]
    stations = list(reader.STATION_COLORS)

    used_threshold = next(iter(diagnostics.values())).get("evidence_threshold", "—")
    st.success(f"解析完了：{len(df)}時点 / 色証拠閾値={used_threshold}")

    # 使用頻度の高いCSVボタンを結果の最上部に置く。
    csv_bytes = df.loc[:, reader.CSV_COLUMNS].to_csv(
        index=False, date_format="%Y/%m/%d %H:%M"
    ).encode("utf-8-sig")
    st.download_button(
        "⬇ CSVをダウンロード（保存後、画像を自動クリア）",
        data=csv_bytes,
        file_name=f"{end_dt:%Y%m%d_%H%M}_viewership.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
        on_click=on_main_csv_download,
    )

    with st.expander("自動較正・品質診断", expanded=False):
        st.json({
            "ソース": f"{reader_path.name} / {reader_hash}",
            "改良バージョン": reader.READER_PATCH_VERSION,
            "0%線（px）": round(calib.y_zero, 2),
            "グラフ上端（px）": round(calib.graph_top, 2),
            "1%あたりpx": round(calib.pixels_per_percent, 3),
            "1分あたりpx": round(calib.pixels_per_minute, 3),
            "推定縦軸上限": round((calib.y_zero - calib.graph_top) / calib.pixels_per_percent, 2),
            "縦軸1目盛り": calib.grid_percent_step,
            "横軸較正": calib.time_axis_method,
            "正時線数": calib.time_line_inliers,
        })
        diag_rows = []
        for station, values in diagnostics.items():
            diag_rows.append({
                "局": station,
                "モード": values.get("range_mode"),
                "使用閾値": values.get("evidence_threshold"),
                "厳格24の証拠率(%)": round(values.get("strict_24_coverage", 0) * 100, 1),
                "緩和35の証拠率(%)": round(values.get("relaxed_35_coverage", 0) * 100, 1),
                "採用した証拠率(%)": round(values["coverage"] * 100, 1),
                "品質": values["quality"],
                "強色証拠での品質": values.get("quality_strict"),
                "最長推定(分)": values.get("max_gap_minutes"),
                "最大1pxジャンプ(%)": values.get("max_step_pp"),
                "平均色コスト": values.get("mean_evidence_cost"),
                "アンカー": values.get("anchor"),
            })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
        if any(v["quality"] == "C" for v in diagnostics.values()):
            st.warning("品質Cの局があります。重ね合わせ表示で確認してください。")
        st.caption("品質A/B/Cは画像上の色証拠に関する評価であり、数値の正確さの保証ではありません。")

    if known is not None:
        check = pd.DataFrame({
            "放送局": stations,
            "画面表示(%)": known,
            "画像復元(%)": [float(df.iloc[-1][station]) for station in stations],
        })
        check["差(pt)"] = (check["画像復元(%)"] - check["画面表示(%)"]).round(2)
        st.subheader(f"終端照合：6局平均絶対誤差 {check['差(pt)'].abs().mean():.3f}pt")
        st.dataframe(check, use_container_width=True)

    if show_overlay:
        with st.expander("解析線の重ね合わせ", expanded=True):
            original = Image.open(BytesIO(result["image_bytes"])).convert("RGB")
            st.image(trace_overlay(original, traces, calib), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["視聴率", "6局シェア", "順位割合", "全データ"])
    with tab1:
        st.line_chart(df.set_index("日時")[stations], height=480)
        st.line_chart(df.set_index("日時")[["6局合計"]], height=280)
    with tab2:
        share_cols = [f"{station}_シェア" for station in stations]
        shares = df.set_index("日時")[share_cols].copy()
        shares.columns = stations
        st.line_chart(shares, height=420)
    with tab3:
        ranks = pd.DataFrame(index=stations)
        for rank in range(1, 7):
            ranks[f"{rank}位"] = [df[f"{s}_順位"].eq(rank).mean() * 100 for s in stations]
        st.bar_chart(ranks, height=420)
        st.dataframe(ranks.style.format("{:.2f}%"), use_container_width=True)
    with tab4:
        st.dataframe(df.loc[:, reader.CSV_COLUMNS], use_container_width=True, height=480)
