"""TVAL now スクリーンショット読取用UI。Streamlit: streamlit run App.py

reader.py は同じフォルダに置くこと。
バージョン9.2の呼び出し形式を保ちつつ、9.2.1の高レンジ自動補正・
追跡オーバーレイ・強弱2種類の色証拠・終端照合を追加。
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

APP_VERSION = "9.2.1-adaptive-range"
EXPECTED_READER_VERSION = "9.2.0-optimized-trace"  # 旧App互換の公開版番号


def load_reader_from_current_source():
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
if reader.READER_VERSION != EXPECTED_READER_VERSION or not hasattr(reader, "READER_PATCH_VERSION"):
    st.error(f"対応する改良版 reader.py が必要です。現在: {reader.READER_VERSION} / patch: {getattr(reader, 'READER_PATCH_VERSION', '不明')}")
    st.stop()

st.title("📺 TVAL now 画像 → 1分刻み視聴率")
st.caption(
    f"UI {APP_VERSION} / reader {reader.READER_VERSION} / "
    f"改良版 {getattr(reader, 'READER_PATCH_VERSION', '不明')} / ソース識別ID {reader_hash}"
)
st.info(
    "CSVの『実測』は画像上で色線の証拠が認められたことを表します。"
    "TVAL nowの原数値を直接取得したものではありません。"
)

col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded = st.file_uploader("グラフ全体のスクリーンショット", type=["png", "jpg", "jpeg", "webp"])
    end_date = st.date_input("『データ更新』の日付", value=None)
    end_time_text = st.text_input("『データ更新』の時刻", value="", placeholder="例: 21:15")
    duration = st.number_input("グラフ表示時間（分）", min_value=30, max_value=360, value=180)
    grid_step = st.number_input(
        "縦軸1目盛り（%）", min_value=0.5, max_value=20.0, value=2.0, step=0.5,
        help="0,2,4,…なら2。0,4,8,…なら4。画像に記載された値を入力してください。"
    )
with col_right:
    mode = st.selectbox(
        "色線解析モード",
        ["auto", "strict", "high-range"],
        format_func=lambda x: {
            "auto": "自動（推奨）", "strict": "従来の厳格24", "high-range": "高レンジ35（強制・検証用）"
        }[x],
    )
    rgb_tolerance = st.slider("カーソルRGB差", 10, 50, 20)
    show_overlay = st.checkbox("抽出経路を元画像に重ねて検証する", value=True)
    known_endpoint = st.text_input(
        "画面上部に表示された6局の数値（任意）",
        placeholder="例: 3.2, 5.7, 6.5, 20.0, 3.8, 3.2",
        help="NHK, 日テレ, テレ朝, TBS, テレ東, フジの順。入力すると21:15などの終端を検証します。"
    )

raw = uploaded.getvalue() if uploaded is not None else None
if raw is not None:
    preview = Image.open(BytesIO(raw)).convert("RGB")
    st.caption(f"画像: {uploaded.name} / {preview.width}×{preview.height} / SHA256: {hashlib.sha256(raw).hexdigest()[:12]}")
    with st.expander("読込画像を表示", expanded=False):
        st.image(preview, use_container_width=True)


def parse_end_values(text):
    if not text.strip():
        return None
    parts = [p.strip().replace("％", "").replace("%", "") for p in text.replace("、", ",").split(",")]
    if len(parts) != 6:
        raise ValueError("画面表示値は、NHK→日テレ→テレ朝→TBS→テレ東→フジの6個を入力してください")
    return [float(p) for p in parts]


def trace_overlay(image, traces, calib):
    """解析された経路を元画像に重ねる。元画像は変更せず新しい画像を作る。"""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    palette = {
        "NHK総合": (70, 70, 70), "日本テレビ": (235, 166, 0),
        "テレビ朝日": (210, 48, 190), "TBS": (0, 88, 240),
        "テレビ東京": (20, 35, 108), "フジテレビ": (215, 35, 35),
    }
    for station, (xs, ys, observed) in traces.items():
        # 画面での確認を目的とするため観測のある線と推定線を同じ色で描く。
        points = [(int(x), int(round(float(y)))) for x, y in zip(xs, ys)]
        if len(points) >= 2:
            draw.line(points, fill=palette[station], width=2)
        # 大きな推定区間の中心だけを黒円で示す。
        inv = np.flatnonzero(~observed)
        if len(inv):
            splits = np.split(inv, np.where(np.diff(inv) != 1)[0] + 1)
            for indices in splits:
                if len(indices) >= max(8, int(calib.pixels_per_minute * 2)):
                    j = int(indices[len(indices) // 2])
                    x, y = int(xs[j]), int(round(float(ys[j])))
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="black")
    return out


if st.button("▶ 解析する", type="primary", use_container_width=True):
    if raw is None or end_date is None:
        st.error("画像と『データ更新』の日付を指定してください。")
        st.stop()
    try:
        parsed_time = datetime.strptime(end_time_text.strip(), "%H:%M").time()
        known = parse_end_values(known_endpoint)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    end_dt = datetime.combine(end_date, parsed_time)
    image = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.asarray(image)

    try:
        df, calib, diagnostics, traces = reader.analyze_image(
            arr=arr,
            end_time=end_dt,
            duration_minutes=int(duration),
            tolerance=int(rgb_tolerance),
            grid_percent_step=float(grid_step),
            evidence_mode=mode,
            return_traces=True,
        )
    except Exception as exc:
        st.error("解析に失敗しました。グラフ全体・0%線・水平線の有無を確認してください。")
        st.exception(exc)
        st.stop()

    stations = list(reader.STATION_COLORS)
    st.success(f"解析完了：{len(df)}時点。実使用閾値={next(iter(diagnostics.values()))['evidence_threshold']}")

    with st.expander("グラフ較正結果", expanded=True):
        st.json({
            "ソース": f"{reader_path.name} / {reader_hash}",
            "改良バージョン": reader.READER_PATCH_VERSION,
            "0%線（px）": round(calib.y_zero, 2),
            "グラフ上端（px）": round(calib.graph_top, 2),
            "1%あたりpx": round(calib.pixels_per_percent, 3),
            "1分あたりpx": round(calib.pixels_per_minute, 3),
            "推定縦軸上限": round((calib.y_zero - calib.graph_top) / calib.pixels_per_percent, 2),
            "縦軸1目盛り": float(grid_step),
            "横軸較正": calib.time_axis_method,
            "正時線数": calib.time_line_inliers,
        })
        if calib.time_axis_method != "hour-lines":
            st.warning("正時線から較正できず、色線の端から横軸を決定しました。誤差に注意してください。")

    diag_rows = []
    for station, d in diagnostics.items():
        diag_rows.append({
            "局": station,
            "モード": d["range_mode"],
            "使用閾値": d["evidence_threshold"],
            "厳格24の証拠率(%)": round(d["strict_24_coverage"] * 100, 1),
            "緩和35の証拠率(%)": round(d["relaxed_35_coverage"] * 100, 1),
            "採用した証拠率(%)": round(d["coverage"] * 100, 1),
            "品質(採用基準)": d["quality"],
            "強色証拠での品質": d["quality_strict"],
            "最長推定(分)": d["max_gap_minutes"],
            "最大1pxジャンプ(%)": d["max_step_pp"],
            "平均色コスト": d["mean_evidence_cost"],
            "アンカー": d["anchor"],
        })
    st.subheader("6局の追跡診断：厳格24と緩和35の両方を表示")
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
    if calib.pixels_per_minute < 8:
        st.warning("横解像度が1分8px未満です。カーソル色や線の交差が誤読される場合があります。元の大きさの画像で比較してください。")
    if any(d["quality"] == "C" for d in diagnostics.values()):
        st.warning("採用した閾値でも品質Cの局があります。経路重ね合わせを確認してください。")
    st.caption("重要：品質A/B/Cは色線の追跡証拠であり、真の視聴率との誤差保証ではありません。")

    if known is not None:
        table = pd.DataFrame({
            "放送局": stations,
            "画面表示(%)": known,
            "画像復元(%)": [float(df.iloc[-1][s]) for s in stations],
        })
        table["差(pt)"] = (table["画像復元(%)"] - table["画面表示(%)"]).round(2)
        mae = table["差(pt)"].abs().mean()
        st.subheader(f"終端の実数照合：6局平均絶対誤差 {mae:.3f}pt")
        st.dataframe(table, use_container_width=True)
        if table["差(pt)"].abs().max() > .3:
            st.warning("終端で0.3ポイント超のずれがあります。CSVを確定する前に重ね合わせを確認してください。")

    if show_overlay:
        with st.expander("元画像と抽出線を重ねて確認（推奨）", expanded=True):
            st.image(trace_overlay(image, traces, calib), use_container_width=True)
            st.caption("色線から離れる区間・別局の線に飛び移る区間がないか、全幅で確認してください。黒点は長い推定区間です。")

    tab1, tab2, tab3, tab4 = st.tabs(["視聴率", "6局シェア", "順位割合", "全データ"])
    with tab1:
        st.line_chart(df.set_index("日時")[stations], height=480)
        st.line_chart(df.set_index("日時")[["6局合計"]], height=280)
    with tab2:
        cols = [f"{s}_シェア" for s in stations]
        share_df = df.set_index("日時")[cols].copy()
        share_df.columns = stations
        st.line_chart(share_df, height=420)
    with tab3:
        rank_rates = pd.DataFrame(index=stations)
        for rank in range(1, 7):
            rank_rates[f"{rank}位"] = [df[f"{s}_順位"].eq(rank).mean() * 100 for s in stations]
        st.bar_chart(rank_rates, height=420)
        st.dataframe(rank_rates.style.format("{:.2f}%"), use_container_width=True)
    with tab4:
        st.dataframe(df, use_container_width=True, height=480)

    st.download_button(
        "⬇ 解析CSVを保存（旧版と同じ27列）",
        data=df.loc[:, reader.CSV_COLUMNS].to_csv(index=False, date_format="%Y/%m/%d %H:%M").encode("utf-8-sig"),
        file_name=f"{end_dt:%Y%m%d_%H%M}_viewership.csv", mime="text/csv", use_container_width=True,
    )
    st.download_button(
        "⬇ 診断表CSVを保存", data=pd.DataFrame(diag_rows).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{end_dt:%Y%m%d_%H%M}_diagnostics.csv", mime="text/csv", use_container_width=True,
    )
