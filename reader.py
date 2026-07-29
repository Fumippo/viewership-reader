from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


READER_VERSION = "6.0.0-grid-time-recovery"

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
    pixels_per_minute: float
    start_time: datetime
    end_time: datetime
    duration_minutes: int


def _group_consecutive(values: np.ndarray | list[int], max_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw in values:
        value = int(raw)
        if not groups or value - groups[-1][-1] > max_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _neutral_gray_mask(
    pixels: np.ndarray,
    *,
    max_spread: int,
    min_brightness: float,
    max_brightness: float,
) -> np.ndarray:
    work = pixels.astype(np.int16)
    spread = work.max(axis=2) - work.min(axis=2)
    brightness = work.mean(axis=2)
    return (
        (spread <= max_spread)
        & (brightness >= min_brightness)
        & (brightness <= max_brightness)
    )


def _find_thin_horizontal_lines(arr: np.ndarray) -> list[int]:
    """
    画像下端近くまで調べ、長い薄灰色の細線を抽出する。
    以前の0.90h打ち切りは廃止。0%線が95%地点でも拾う。
    """
    h, w, _ = arr.shape
    x0 = int(w * 0.08)
    x1 = int(w * 0.95)

    crop = arr[:, x0:x1]
    mask = _neutral_gray_mask(
        crop,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    row_score = mask.mean(axis=1)

    candidates = np.where(row_score >= 0.65)[0]
    groups = _group_consecutive(candidates, max_gap=1)

    # グラフ線は通常1～2px。UIカードなどの太い帯は除外。
    thin_lines = [
        int(round(np.mean(group)))
        for group in groups
        if 1 <= len(group) <= 4
        and int(h * 0.18) <= int(round(np.mean(group))) <= int(h * 0.99)
    ]

    if len(thin_lines) < 5:
        raise RuntimeError(
            "水平グリッド線を十分に検出できませんでした。"
            "グラフ全体と0%線が入った画像を使ってください。"
        )

    return thin_lines


def _select_regular_grid(lines: list[int], image_height: int) -> tuple[list[int], float]:
    """
    UI由来の別水平線が混ざっても、最長の等間隔列だけを採用する。
    """
    ys = np.array(sorted(set(lines)), dtype=float)
    gap_candidates: list[float] = []

    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            difference = ys[j] - ys[i]
            for steps in range(1, 10):
                gap = difference / steps
                if 45.0 <= gap <= 220.0:
                    gap_candidates.append(float(gap))

    best_score = -1e18
    best_values: list[float] | None = None

    for gap in gap_candidates:
        tolerance = max(3.0, gap * 0.035)

        for anchor in ys:
            step_numbers = np.round((ys - anchor) / gap)
            residuals = np.abs(ys - (anchor + step_numbers * gap))
            matched_steps = np.unique(
                step_numbers[residuals <= tolerance].astype(int)
            )

            if len(matched_steps) < 5:
                continue

            step_groups = _group_consecutive(matched_steps, max_gap=1)
            longest_steps = max(step_groups, key=len)

            values: list[float] = []
            residual_sum = 0.0
            for step in longest_steps:
                target = anchor + step * gap
                index = int(np.argmin(np.abs(ys - target)))
                values.append(float(ys[index]))
                residual_sum += abs(float(ys[index]) - target)

            if len(values) < 5:
                continue

            span = max(values) - min(values)
            score = len(values) * 1000.0 + span - residual_sum

            if score > best_score:
                best_score = score
                best_values = values

    if best_values is None or len(best_values) < 5:
        raise RuntimeError("規則的な水平グリッドを特定できませんでした。")

    grid = sorted(set(int(round(value)) for value in best_values))
    gap = float(np.median(np.diff(grid)))

    # 2px程度の描画揺れは許すが、崩れた列は拒否。
    if np.max(np.abs(np.diff(grid) - gap)) > max(4.0, gap * 0.06):
        raise RuntimeError("水平グリッドの間隔が不規則です。")

    return grid, gap


def _find_rough_graph_x_bounds(arr: np.ndarray, grid: list[int]) -> tuple[int, int]:
    """
    選んだ全水平線で共通して灰色になっている横範囲をグラフ領域とする。
    """
    rows = arr[np.array(grid, dtype=int)]
    mask = _neutral_gray_mask(
        rows,
        max_spread=25,
        min_brightness=190,
        max_brightness=253,
    )
    column_score = mask.mean(axis=0)

    candidates = np.where(column_score >= 0.75)[0]
    groups = _group_consecutive(candidates, max_gap=6)

    if not groups:
        raise RuntimeError("グラフ左右端を検出できませんでした。")

    longest = max(groups, key=lambda group: group[-1] - group[0])
    left, right = longest[0], longest[-1]

    if right - left < arr.shape[1] * 0.45:
        raise RuntimeError("検出したグラフ幅が狭すぎます。")

    return int(left), int(right)


def _find_vertical_hour_lines(
    arr: np.ndarray,
    graph_top: int,
    graph_bottom: int,
    rough_left: int,
    rough_right: int,
) -> list[int]:
    """
    1時間刻みの縦破線を抽出する。
    選択カーソルの実線は縦方向スコアが高すぎるため除外する。
    """
    roi = arr[graph_top:graph_bottom + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    column_score = mask.mean(axis=0)

    x_axis = np.arange(arr.shape[1])
    candidates = np.where(
        (column_score >= 0.35)
        & (x_axis >= rough_left)
        & (x_axis <= rough_right)
    )[0]
    groups = _group_consecutive(candidates, max_gap=2)

    peaks: list[int] = []
    for group in groups:
        center = int(round(np.mean(group)))
        score = float(np.max(column_score[group]))

        # 破線は2px前後。太いUI帯と連続実線カーソルを除外。
        if len(group) <= 6 and 0.38 <= score <= 0.76:
            peaks.append(center)

    return sorted(set(peaks))


def _select_hour_progression(peaks: list[int]) -> list[int]:
    if len(peaks) < 2:
        return []

    best_score = -1e18
    best: list[int] = []

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            gap = peaks[j] - peaks[i]
            if gap < 200:
                continue

            tolerance = max(4.0, gap * 0.015)
            matched: list[int] = []

            for x in peaks:
                step = round((x - peaks[i]) / gap)
                target = peaks[i] + step * gap
                if abs(x - target) <= tolerance:
                    matched.append(x)

            matched = sorted(set(matched))
            if len(matched) < 2:
                continue

            score = len(matched) * 1000.0 + (matched[-1] - matched[0])
            if score > best_score:
                best_score = score
                best = matched

    return best


def _calibrate_time_axis(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
    end_time: datetime,
    duration_minutes: int,
) -> tuple[float, float, float]:
    """
    水平グリッドの左右端ではなく、1時間刻みの縦破線から時間軸を較正する。
    これによりグラフ内側の余白を時間として数える誤差を防ぐ。
    """
    peaks = _find_vertical_hour_lines(
        arr,
        grid[0],
        grid[-1],
        rough_left,
        rough_right,
    )
    hours = _select_hour_progression(peaks)

    if len(hours) < 2:
        # 縦破線が取れない画像だけ、粗い左右端へフォールバック。
        pixels_per_minute = (rough_right - rough_left) / duration_minutes
        return float(rough_left), float(rough_right), float(pixels_per_minute)

    hour_spacing = float(np.median(np.diff(hours)))
    pixels_per_minute = hour_spacing / 60.0

    rightmost_hour_x = float(hours[-1])
    end_x = rightmost_hour_x + end_time.minute * pixels_per_minute

    # ちょうど00分でカーソルと重なり、最新の時線が除外された場合の補正。
    if end_x < rough_right - 0.55 * hour_spacing:
        end_x += hour_spacing

    start_x = end_x - duration_minutes * pixels_per_minute

    allowed_margin = hour_spacing * 0.15
    if (
        start_x < rough_left - allowed_margin
        or end_x > rough_right + allowed_margin
    ):
        raise RuntimeError("縦の時刻目盛りから時間軸を較正できませんでした。")

    return float(start_x), float(end_x), float(pixels_per_minute)


def _station_color_mask(
    roi: np.ndarray,
    station: str,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    work = roi.astype(np.int16)
    target = np.array(target_rgb, dtype=np.int16)
    distance = np.max(np.abs(work - target), axis=2)

    if station == "NHK総合":
        # NHK灰色と薄い水平グリッドを分離する。
        spread = work.max(axis=2) - work.min(axis=2)
        brightness = work.mean(axis=2)
        return (
            (distance <= 30)
            & (brightness < 218)
            & (spread <= 40)
        )

    return distance <= tolerance


def _trace_station(
    arr: np.ndarray,
    station: str,
    color: Tuple[int, int, int],
    rough_left: int,
    rough_right: int,
    graph_top: int,
    graph_bottom: int,
    pixels_per_percent: float,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """
    色線を左から追う。

    重要:
    - 大きな縦カーソル列を除外
    - 最初の孤立1pxを開始点にしない
    - 不自然な単発大ジャンプは捨てる
    - ただし前回のように永久停止せず、元の線が戻れば追跡再開
    """
    roi = arr[
        graph_top:graph_bottom + 1,
        rough_left:rough_right + 1,
    ]
    mask = _station_color_mask(roi, station, color, tolerance)

    # 選択カーソルの縦線など、縦一列に大量の同色がある列を除外。
    column_counts = mask.sum(axis=0)
    vertical_limit = max(80, int(mask.shape[0] * 0.15))
    mask[:, column_counts > vertical_limit] = False

    xs = np.arange(rough_left, rough_right + 1)
    ys = np.full(len(xs), np.nan)

    previous_y: float | None = None
    rejected_jumps = 0
    max_single_column_jump = max(70.0, pixels_per_percent * 1.5)

    for index in range(mask.shape[1]):
        positions = np.where(mask[:, index])[0]
        if len(positions) == 0:
            continue

        groups = _group_consecutive(positions, max_gap=1)
        candidates = [
            (
                float(np.median(group)) + graph_top,
                len(group),
            )
            for group in groups
        ]

        if previous_y is None:
            # 孤立した1pxノイズで開始しない。
            strong = [candidate for candidate in candidates if candidate[1] >= 2]
            if not strong:
                continue
            chosen_y = max(strong, key=lambda candidate: candidate[1])[0]
        else:
            chosen = min(
                candidates,
                key=lambda candidate:
                    abs(candidate[0] - previous_y)
                    - min(candidate[1], 20) * 0.1,
            )

            # 1列だけ数百px飛ぶNHKカーソル周辺ノイズなどを拒否。
            # 拒否してもprevious_yは維持し、正しい線が戻れば復帰できる。
            if abs(chosen[0] - previous_y) > max_single_column_jump:
                rejected_jumps += 1
                continue

            chosen_y = chosen[0]

        ys[index] = chosen_y
        previous_y = chosen_y

    good = ~np.isnan(ys)
    coverage = float(good.mean())

    missing_groups = _group_consecutive(np.where(~good)[0], max_gap=1)
    max_gap_pixels = max((len(group) for group in missing_groups), default=0)

    # 壊れた結果を完成品として出さない安全装置。
    if coverage < 0.85:
        raise RuntimeError(
            f"{station}の線検出率が{coverage * 100:.1f}%しかありません。"
            "この画像では安全に数値化できません。"
        )

    if max_gap_pixels > 60:
        raise RuntimeError(
            f"{station}で{max_gap_pixels}pxの連続欠損が発生しました。"
            "長距離補間を避けるため解析を中止しました。"
        )

    interpolated = np.interp(xs, xs[good], ys[good])

    diagnostics: dict[str, float | int] = {
        "coverage": coverage,
        "max_gap_pixels": int(max_gap_pixels),
        "rejected_jumps": int(rejected_jumps),
    }
    return xs, interpolated, diagnostics


def analyze_image(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
    tolerance: int = 42,
) -> tuple[pd.DataFrame, Calibration, dict[str, dict[str, float | int]]]:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB画像を指定してください。")

    thin_lines = _find_thin_horizontal_lines(arr)
    grid, grid_gap = _select_regular_grid(thin_lines, arr.shape[0])

    graph_top = int(grid[0])
    y_zero = int(grid[-1])
    pixels_per_percent = grid_gap / 2.0

    rough_left, rough_right = _find_rough_graph_x_bounds(arr, grid)
    start_x, end_x, pixels_per_minute = _calibrate_time_axis(
        arr,
        grid,
        rough_left,
        rough_right,
        end_time,
        duration_minutes,
    )

    calibration = Calibration(
        graph_left=start_x,
        graph_right=end_x,
        graph_top=float(graph_top),
        graph_bottom=float(y_zero),
        y_zero=float(y_zero),
        pixels_per_percent=float(pixels_per_percent),
        pixels_per_minute=float(pixels_per_minute),
        start_time=end_time - timedelta(minutes=duration_minutes),
        end_time=end_time,
        duration_minutes=duration_minutes,
    )

    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}

    for station, color in STATION_COLORS.items():
        xs, ys, station_diagnostics = _trace_station(
            arr,
            station,
            color,
            rough_left,
            rough_right,
            graph_top,
            y_zero,
            pixels_per_percent,
            tolerance,
        )
        traces[station] = (xs, ys)
        diagnostics[station] = station_diagnostics

    times = [
        calibration.start_time + timedelta(minutes=minute)
        for minute in range(duration_minutes + 1)
    ]
    df = pd.DataFrame({"日時": times})

    for station, (xs, ys) in traces.items():
        values: list[float] = []

        for minute in range(duration_minutes + 1):
            x = start_x + minute * pixels_per_minute
            y = float(np.interp(x, xs, ys))
            rating = (y_zero - y) / pixels_per_percent
            values.append(round(max(0.0, rating), 1))

        df[station] = values

    stations = list(STATION_COLORS)
    df["6局合計"] = df[stations].sum(axis=1).round(1)
    df["その他5局合計"] = (df["6局合計"] - df["NHK総合"]).round(1)

    for station in stations:
        df[f"{station}_シェア"] = (
            df[station]
            / df["6局合計"].replace(0, np.nan)
            * 100
        ).round(2)

    ranks = df[stations].rank(axis=1, ascending=False, method="min")
    for station in stations:
        df[f"{station}_順位"] = ranks[station].astype("Int64")

    return df, calibration, diagnostics
