from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


STATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "NHK総合": (182, 188, 195),
    "日本テレビ": (254, 216, 120),
    "テレビ朝日": (246, 162, 228),
    "TBS": (100, 148, 240),
    "テレビ東京": (89, 113, 175),
    "フジテレビ": (243, 124, 124),
}


@dataclass
class Calibration:
    graph_left: float
    graph_right: float
    y_zero: float
    pixels_per_percent: float
    end_time: datetime
    duration_minutes: int = 180

    @property
    def pixels_per_minute(self) -> float:
        return (self.graph_right - self.graph_left) / self.duration_minutes

    @property
    def start_time(self) -> datetime:
        return self.end_time - timedelta(minutes=self.duration_minutes)


def find_color_x_range(arr: np.ndarray) -> Tuple[int, int]:
    xs_all: List[int] = []
    h, w, _ = arr.shape

    # 上部の局カードを避け、下半分のグラフ部分だけ見る
    y0 = int(h * 0.28)
    y1 = int(h * 0.90)

    for rgb in STATION_COLORS.values():
        ys, xs = np.where(np.all(arr[y0:y1, :, :] == rgb, axis=2))
        if len(xs):
            xs_all.extend(xs.tolist())

    if not xs_all:
        raise RuntimeError("局カラーの折れ線を検出できませんでした。")

    return min(xs_all), max(xs_all)


def detect_horizontal_gridlines(arr: np.ndarray, x0: int, x1: int) -> List[int]:
    h, w, _ = arr.shape
    # 薄い灰色の水平グリッドを検出
    crop = arr[int(h * 0.25):int(h * 0.90), x0:x1 + 1, :]
    grayness = crop.max(axis=2) - crop.min(axis=2)
    brightness = crop.mean(axis=2)

    # ほぼ無彩色・明るめのピクセル
    mask = (grayness <= 5) & (brightness >= 210) & (brightness <= 250)
    row_score = mask.mean(axis=1)

    candidates = np.where(row_score > 0.35)[0] + int(h * 0.25)
    if len(candidates) == 0:
        raise RuntimeError("水平グリッド線を検出できませんでした。")

    # 連続行をまとめる
    groups: List[List[int]] = []
    for y in candidates:
        if not groups or y - groups[-1][-1] > 2:
            groups.append([int(y)])
        else:
            groups[-1].append(int(y))

    centers = [int(round(sum(g) / len(g))) for g in groups]
    return centers


def estimate_y_scale(gridlines: List[int]) -> Tuple[float, float]:
    if len(gridlines) < 3:
        raise RuntimeError("縦軸較正に必要な水平線が不足しています。")

    gridlines = sorted(gridlines)
    diffs = np.diff(gridlines)

    # 最頻に近い間隔を採用
    median_gap = float(np.median(diffs))
    good = diffs[(diffs > median_gap * 0.7) & (diffs < median_gap * 1.3)]
    gap = float(np.median(good)) if len(good) else median_gap

    # TVALの縦目盛りは通常2ポイント刻み
    pixels_per_percent = gap / 2.0
    y_zero = float(max(gridlines))
    return y_zero, pixels_per_percent


def parse_end_time(value: str) -> datetime:
    formats = [
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%m/%d %H:%M",
        "%H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%m/%d %H:%M":
                dt = dt.replace(year=datetime.now().year)
            elif fmt == "%H:%M":
                now = datetime.now()
                dt = dt.replace(year=now.year, month=now.month, day=now.day)
            return dt
        except ValueError:
            pass
    raise ValueError("終了時刻は 2026/07/29 19:00 の形式で入力してください。")


def trace_station_line(
    arr: np.ndarray,
    rgb: Tuple[int, int, int],
    x0: int,
    x1: int,
    y_min: int,
    y_max: int,
    tolerance: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    x_grid = np.arange(x0, x1 + 1)
    centers = np.full(len(x_grid), np.nan)

    target = np.array(rgb, dtype=int)

    for i, x in enumerate(x_grid):
        col = arr[y_min:y_max + 1, x, :].astype(int)
        if tolerance == 0:
            mask = np.all(col == target, axis=1)
        else:
            mask = np.max(np.abs(col - target), axis=1) <= tolerance

        ys = np.where(mask)[0] + y_min
        if len(ys):
            centers[i] = float(np.median(ys))

    good = ~np.isnan(centers)
    if good.sum() < 10:
        raise RuntimeError(f"色 {rgb} の線を十分に追跡できませんでした。")

    interp = np.interp(x_grid, x_grid[good], centers[good])
    return x_grid, interp


def build_dataframe(
    arr: np.ndarray,
    calibration: Calibration,
    tolerance: int = 0,
) -> pd.DataFrame:
    x0 = int(round(calibration.graph_left))
    x1 = int(round(calibration.graph_right))
    h = arr.shape[0]

    # グラフ上端を水平グリッド列から逆算する。
    # 0%線から2ポイント刻みで上へたどり、実在する最上段だけを採用する。
    gridlines = detect_horizontal_gridlines(arr, x0, x1)
    step = calibration.pixels_per_percent * 2.0
    graph_top = calibration.y_zero
    expected = calibration.y_zero - step
    tolerance_px = max(4.0, step * 0.08)

    while expected >= 0:
        nearest = min(gridlines, key=lambda y: abs(y - expected))
        if abs(nearest - expected) <= tolerance_px:
            graph_top = float(nearest)
            expected -= step
        else:
            break

    # 上部の局カードを完全に検索対象外にする。
    y_min = max(0, int(round(graph_top - 6)))
    y_max = min(h - 1, int(round(calibration.y_zero + 10)))

    traces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for station, rgb in STATION_COLORS.items():
        traces[station] = trace_station_line(
            arr, rgb, x0, x1, y_min, y_max, tolerance=tolerance
        )

    times = [
        calibration.start_time + timedelta(minutes=i)
        for i in range(calibration.duration_minutes + 1)
    ]

    df = pd.DataFrame({"日時": times})

    for station, (x_grid, y_values) in traces.items():
        ratings: List[float] = []
        for minute in range(calibration.duration_minutes + 1):
            x = calibration.graph_left + minute * calibration.pixels_per_minute
            y = float(np.interp(x, x_grid, y_values))
            rating = (calibration.y_zero - y) / calibration.pixels_per_percent
            ratings.append(round(max(0.0, rating), 1))
        df[station] = ratings

    stations = list(STATION_COLORS)
    df["6局合計"] = df[stations].sum(axis=1).round(1)
    df["その他5局合計"] = (df["6局合計"] - df["NHK総合"]).round(1)

    for station in stations:
        df[f"{station}_シェア"] = (
            df[station] / df["6局合計"].replace(0, np.nan) * 100
        ).round(2)

    ranks = df[stations].rank(axis=1, ascending=False, method="min")
    for station in stations:
        df[f"{station}_順位"] = ranks[station].astype("Int64")

    return df


def save_graphs(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    stations = list(STATION_COLORS)

    def setup_time_axis() -> None:
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

    plt.figure(figsize=(13, 6))
    for station in stations:
        plt.plot(df["日時"], df[station], linewidth=1.8, label=station)
    plt.title("各局の視聴率推移")
    plt.xlabel("時刻")
    plt.ylabel("視聴率（%）")
    plt.legend(ncol=3)
    setup_time_axis()
    plt.savefig(out_dir / f"{prefix}_all_stations.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["日時"], df["6局合計"], linewidth=2)
    plt.title("6局合計視聴率")
    plt.xlabel("時刻")
    plt.ylabel("合計視聴率（%）")
    setup_time_axis()
    plt.savefig(out_dir / f"{prefix}_total.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["日時"], df["NHK総合"], linewidth=2, label="NHK総合")
    plt.plot(df["日時"], df["その他5局合計"], linewidth=2, label="その他5局合計")
    plt.title("NHK総合 vs その他5局合計")
    plt.xlabel("時刻")
    plt.ylabel("視聴率（%）")
    plt.legend()
    setup_time_axis()
    plt.savefig(out_dir / f"{prefix}_nhk_vs_others.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(13, 6))
    for station in stations:
        plt.plot(
            df["日時"],
            df[f"{station}_シェア"],
            linewidth=1.8,
            label=station,
        )
    plt.title("各局の6局内シェア率推移")
    plt.xlabel("時刻")
    plt.ylabel("シェア率（%）")
    plt.ylim(0, 50)
    plt.legend(ncol=3)
    setup_time_axis()
    plt.savefig(out_dir / f"{prefix}_share_trend.png", dpi=200, bbox_inches="tight")
    plt.close()

    ranks = pd.DataFrame(
        {station: df[f"{station}_順位"] for station in stations}
    )
    rank_rates = pd.DataFrame(index=stations, columns=range(1, 7), dtype=float)
    for station in stations:
        for rank in range(1, 7):
            rank_rates.loc[station, rank] = (ranks[station] == rank).mean() * 100

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(stations))
    for rank in range(1, 7):
        vals = rank_rates[rank].values
        plt.bar(stations, vals, bottom=bottom, label=f"{rank}位")
        bottom += vals
    plt.title("各局の順位獲得割合")
    plt.xlabel("局")
    plt.ylabel("割合（%）")
    plt.ylim(0, 100)
    plt.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_rank_share.png", dpi=200, bbox_inches="tight")
    plt.close()

    rank_rates.columns = [f"{c}位率" for c in rank_rates.columns]
    rank_rates.insert(0, "局", rank_rates.index)
    rank_rates.reset_index(drop=True).to_csv(
        out_dir / f"{prefix}_rank_share.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TVAL now のスクリーンショットから1分刻み視聴率を推定します。"
    )
    parser.add_argument("image", help="スクリーンショット画像のパス")
    parser.add_argument(
        "--end-time",
        required=True,
        help='グラフ右端の時刻。例: "2026/07/29 19:00"',
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=180,
        help="グラフの表示時間（分）。通常は180",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="色一致の許容差。検出できない場合は2〜5を試してください。",
    )
    parser.add_argument(
        "--out-dir",
        default="tval_output",
        help="出力先フォルダ",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.array(Image.open(image_path).convert("RGB"))

    color_left, color_right = find_color_x_range(arr)

    # グラフ左右端。色線の出現範囲には丸印などが混じるため少し内側に補正
    graph_left = float(color_left)
    graph_right = float(color_right)

    gridlines = detect_horizontal_gridlines(
        arr,
        int(graph_left),
        int(graph_right),
    )
    y_zero, pixels_per_percent = estimate_y_scale(gridlines)

    end_time = parse_end_time(args.end_time)

    calibration = Calibration(
        graph_left=graph_left,
        graph_right=graph_right,
        y_zero=y_zero,
        pixels_per_percent=pixels_per_percent,
        end_time=end_time,
        duration_minutes=args.duration,
    )

    print("自動較正結果")
    print(f"  グラフ左端: {calibration.graph_left:.1f}px")
    print(f"  グラフ右端: {calibration.graph_right:.1f}px")
    print(f"  0%線: {calibration.y_zero:.1f}px")
    print(f"  1%あたり: {calibration.pixels_per_percent:.2f}px")
    print(f"  開始時刻: {calibration.start_time}")
    print(f"  終了時刻: {calibration.end_time}")

    df = build_dataframe(arr, calibration, tolerance=args.tolerance)

    prefix = end_time.strftime("%Y%m%d_%H%M")
    csv_path = out_dir / f"{prefix}_viewership.csv"
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y/%m/%d %H:%M",
    )

    save_graphs(df, out_dir, prefix)

    print("\n出力完了")
    print(f"  CSV: {csv_path}")
    print(f"  グラフ: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
