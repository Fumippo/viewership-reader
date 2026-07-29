from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


STATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "NHK総合": (182, 188, 195),
    "日本テレビ": (254, 216, 120),
    "テレビ朝日": (246, 162, 228),
    "TBS": (100, 148, 240),
    "テレビ東京": (89, 113, 175),
    "フジテレビ": (243, 124, 124),
}


@dataclass(frozen=True)
class Calibration:
    graph_left: float
    graph_right: float
    graph_top: float
    graph_bottom: float
    y_zero: float
    pixels_per_percent: float
    start_time: datetime
    end_time: datetime
    duration_minutes: int


def _group_consecutive(values: np.ndarray, max_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw in values:
        value = int(raw)
        if not groups or value - groups[-1][-1] > max_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _find_horizontal_lines(arr: np.ndarray) -> list[int]:
    """
    長い明灰色の水平線だけを拾う。
    上部カードや文字は横幅が短いため除外される。
    """
    h, w, _ = arr.shape
    y0 = int(h * 0.25)
    y1 = int(h * 0.90)
    crop = arr[y0:y1].astype(np.int16)

    spread = crop.max(axis=2) - crop.min(axis=2)
    brightness = crop.mean(axis=2)

    gray = (spread <= 14) & (brightness >= 205) & (brightness <= 252)
    min_width = int(w * 0.55)

    candidates: list[int] = []
    for local_y in range(gray.shape[0]):
        xs = np.where(gray[local_y])[0]
        if len(xs) < min_width:
            continue

        groups = _group_consecutive(xs, max_gap=3)
        longest = max(groups, key=len)
        if longest[-1] - longest[0] + 1 >= min_width:
            candidates.append(local_y + y0)

    if not candidates:
        raise RuntimeError("水平グリッド線を検出できませんでした。")

    grouped = _group_consecutive(np.array(candidates), max_gap=3)
    return [int(round(np.mean(group))) for group in grouped]


def _find_graph_x_bounds(arr: np.ndarray, horizontal_lines: list[int]) -> tuple[int, int]:
    """
    水平線の実際の連続範囲からグラフ左右端を決める。
    """
    h, w, _ = arr.shape
    lefts: list[int] = []
    rights: list[int] = []

    for y in horizontal_lines:
        row = arr[y].astype(np.int16)
        spread = row.max(axis=1) - row.min(axis=1)
        brightness = row.mean(axis=1)
        mask = (spread <= 14) & (brightness >= 205) & (brightness <= 252)
        xs = np.where(mask)[0]
        if len(xs) < int(w * 0.50):
            continue

        groups = _group_consecutive(xs, max_gap=3)
        longest = max(groups, key=len)
        if longest[-1] - longest[0] + 1 >= int(w * 0.50):
            lefts.append(longest[0])
            rights.append(longest[-1])

    if not lefts:
        raise RuntimeError("グラフ左右端を検出できませんでした。")

    return int(round(np.median(lefts))), int(round(np.median(rights)))


def _estimate_vertical_scale(horizontal_lines: list[int]) -> tuple[float, float, float]:
    """
    TVALの横線は通常2ポイント刻み。
    最下段を0%、最上段をグラフ上端として扱う。
    """
    ys = np.array(sorted(horizontal_lines), dtype=float)
    if len(ys) < 3:
        raise RuntimeError("水平グリッド線が不足しています。")

    diffs = np.diff(ys)
    diffs = diffs[diffs >= 8]
    if len(diffs) < 2:
        raise RuntimeError("縦軸間隔を推定できませんでした。")

    gap0 = float(np.median(diffs))
    regular = diffs[(diffs >= gap0 * 0.75) & (diffs <= gap0 * 1.25)]
    gap = float(np.median(regular)) if len(regular) else gap0

    y_zero = float(max(ys))
    selected = [y_zero]
    current = y_zero

    while True:
        target = current - gap
        nearby = ys[np.abs(ys - target) <= max(5.0, gap * 0.15)]
        if len(nearby) == 0:
            break
        chosen = float(nearby[np.argmin(np.abs(nearby - target))])
        selected.append(chosen)
        current = chosen

    if len(selected) < 3:
        raise RuntimeError("規則的な水平グリッドを特定できませんでした。")

    graph_top = min(selected)
    pixels_per_percent = gap / 2.0
    return graph_top, y_zero, pixels_per_percent


def build_calibration(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
) -> Calibration:
    lines = _find_horizontal_lines(arr)
    graph_left, graph_right = _find_graph_x_bounds(arr, lines)
    graph_top, y_zero, pixels_per_percent = _estimate_vertical_scale(lines)

    return Calibration(
        graph_left=float(graph_left),
        graph_right=float(graph_right),
        graph_top=float(graph_top),
        graph_bottom=float(y_zero),
        y_zero=float(y_zero),
        pixels_per_percent=float(pixels_per_percent),
        start_time=end_time - timedelta(minutes=duration_minutes),
        end_time=end_time,
        duration_minutes=duration_minutes,
    )


def _color_mask(
    pixels: np.ndarray,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    target = np.array(target_rgb, dtype=np.int16)
    diff = np.abs(pixels.astype(np.int16) - target)
    return np.max(diff, axis=1) <= tolerance


def _trace_one_station(
    arr: np.ndarray,
    color: Tuple[int, int, int],
    calibration: Calibration,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x0 = int(round(calibration.graph_left))
    x1 = int(round(calibration.graph_right))
    y0 = max(0, int(round(calibration.graph_top)) - 3)
    y1 = min(arr.shape[0] - 1, int(round(calibration.graph_bottom)) + 3)

    xs = np.arange(x0, x1 + 1)
    ys_out = np.full(len(xs), np.nan)
    previous: float | None = None
    observed = 0

    for i, x in enumerate(xs):
        column = arr[y0:y1 + 1, x, :]
        raw_ys = np.where(_color_mask(column, color, tolerance))[0] + y0
        if not len(raw_ys):
            continue

        groups = _group_consecutive(raw_ys, max_gap=1)

        # カーソル丸や大きな色面を捨て、細い線だけ残す。
        candidates = [g for g in groups if 1 <= len(g) <= 16]
        if not candidates:
            continue

        centers = [float(np.median(g)) for g in candidates]

        if previous is None:
            chosen = float(np.median(centers))
        else:
            chosen = min(centers, key=lambda y: abs(y - previous))
            # 1列で極端に飛ぶ候補は捨てる。
            max_jump = max(18.0, (y1 - y0) * 0.05)
            if abs(chosen - previous) > max_jump:
                continue

        ys_out[i] = chosen
        previous = chosen
        observed += 1

    good = ~np.isnan(ys_out)
    coverage = observed / len(xs)

    if good.sum() < max(30, int(len(xs) * 0.08)):
        raise RuntimeError(
            f"局色 {color} の線を十分に検出できませんでした。"
            "色許容差を上げるか、元画像を使ってください。"
        )

    # 実測点の間だけ補間。
    interpolated = np.interp(xs, xs[good], ys_out[good])
    return xs, interpolated, coverage


def analyze_image(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
    tolerance: int = 24,
) -> tuple[pd.DataFrame, Calibration, dict[str, float]]:
    calibration = build_calibration(
        arr=arr,
        end_time=end_time,
        duration_minutes=duration_minutes,
    )

    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coverage: dict[str, float] = {}

    for station, color in STATION_COLORS.items():
        xs, ys, station_coverage = _trace_one_station(
            arr=arr,
            color=color,
            calibration=calibration,
            tolerance=tolerance,
        )
        traces[station] = (xs, ys)
        coverage[station] = station_coverage

    times = [
        calibration.start_time + timedelta(minutes=minute)
        for minute in range(duration_minutes + 1)
    ]
    df = pd.DataFrame({"日時": times})

    for station, (xs, ys) in traces.items():
        ratings: list[float] = []
        for minute in range(duration_minutes + 1):
            x = (
                calibration.graph_left
                + minute
                * (calibration.graph_right - calibration.graph_left)
                / duration_minutes
            )
            y = float(np.interp(x, xs, ys))
            rating = (calibration.y_zero - y) / calibration.pixels_per_percent
            ratings.append(round(max(0.0, rating), 1))
        df[station] = ratings

    stations = list(STATION_COLORS.keys())
    df["6局合計"] = df[stations].sum(axis=1).round(1)
    df["その他5局合計"] = (df["6局合計"] - df["NHK総合"]).round(1)

    for station in stations:
        df[f"{station}_シェア"] = (
            df[station] / df["6局合計"].replace(0, np.nan) * 100
        ).round(2)

    ranks = df[stations].rank(axis=1, ascending=False, method="min")
    for station in stations:
        df[f"{station}_順位"] = ranks[station].astype("Int64")

    return df, calibration, coverage
