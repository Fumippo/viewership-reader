from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


READER_VERSION = "8.1.0-regression-tested"

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
    time_axis_method: str = "unknown"
    time_line_inliers: int = 0


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
    上側の水平グリッド線からグラフ左右端を推定する。

    全水平線の共通部分を取る旧方式は、下側の線が番組線で分断されると
    右端を途中で切ってしまうため廃止。
    """
    h, w, _ = arr.shape
    spans: list[tuple[int, int]] = []

    # 上側は折れ線の交差が少なく、水平線が最も長く残りやすい。
    rows_to_use = grid[: min(3, len(grid))]

    for y in rows_to_use:
        row = arr[y:y + 1]
        mask = _neutral_gray_mask(
            row,
            max_spread=25,
            min_brightness=190,
            max_brightness=253,
        )[0]

        xs = np.where(mask)[0]
        if len(xs) == 0:
            continue

        groups = _group_consecutive(xs, max_gap=18)
        long_groups = [
            group for group in groups
            if group[-1] - group[0] + 1 >= max(80, int(w * 0.05))
        ]
        if not long_groups:
            continue

        left = min(group[0] for group in long_groups)
        right = max(group[-1] for group in long_groups)

        if right - left >= w * 0.55:
            spans.append((left, right))

    if not spans:
        raise RuntimeError("グラフ左右端を検出できませんでした。")

    left = int(round(np.median([span[0] for span in spans])))
    right = int(round(np.median([span[1] for span in spans])))

    if right - left < w * 0.55:
        raise RuntimeError("検出したグラフ幅が狭すぎます。")

    return left, right



def _find_vertical_hour_lines(
    arr: np.ndarray,
    graph_top: int,
    graph_bottom: int,
    rough_left: int,
    rough_right: int,
) -> list[int]:
    """
    縦破線の局所ピークを抽出する。

    太い左端UIや選択カーソルを一本の時線として誤認しない。
    """
    roi = arr[graph_top:graph_bottom + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=20,
        min_brightness=195,
        max_brightness=253,
    )
    column_score = mask.mean(axis=0)

    search_left = max(0, rough_left - 60)
    search_right = min(arr.shape[1] - 1, int(arr.shape[1] * 0.98))

    candidates: list[tuple[int, float]] = []
    for x in range(max(1, search_left), min(len(column_score) - 2, search_right) + 1):
        score = float(column_score[x])
        if not (0.30 <= score <= 0.79):
            continue
        if score >= column_score[x - 1] and score >= column_score[x + 1]:
            candidates.append((x, score))

    selected: list[tuple[int, float]] = []
    for x, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(x - old_x) > 20 for old_x, _ in selected):
            selected.append((x, score))

    return sorted(x for x, _ in selected)

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





def _match_time_axis_model(
    peaks: list[int],
    expected_offsets: list[int],
    rough_left: int,
    rough_right: int,
    duration_minutes: int,
    image_width: int,
) -> tuple[float, float, float, list[tuple[int, float]], list[int]]:
    """
    外れ値を含む縦線候補から、正時線だけをRANSAC風に選ぶ。

    戻り値:
    - start_x
    - end_x
    - pixels_per_minute
    - (正時オフセット分, 検出x) の対応
    - 外れ値として除外した縦線
    """
    if len(peaks) < 2:
        raise RuntimeError("縦の時刻目盛りを2本以上検出できませんでした。")
    if len(expected_offsets) < 2:
        raise RuntimeError("表示期間内の正時候補が不足しています。")

    rough_ppm = (rough_right - rough_left) / float(duration_minutes)
    rough_hour_spacing = rough_ppm * 60.0

    max_hour_steps = max(1, int(np.ceil(duration_minutes / 60.0)))
    spacing_candidates: list[float] = [rough_hour_spacing]

    # 2本の候補線が何時間離れているかを1～表示時間分まで試す。
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            difference = float(peaks[j] - peaks[i])
            for hour_steps in range(1, max_hour_steps + 1):
                spacing = difference / hour_steps
                if rough_hour_spacing * 0.55 <= spacing <= rough_hour_spacing * 1.45:
                    spacing_candidates.append(spacing)

    best: dict[str, object] | None = None

    for hour_spacing in spacing_candidates:
        ppm = hour_spacing / 60.0
        tolerance = max(7.0, hour_spacing * 0.035)

        for peak in peaks:
            for offset in expected_offsets:
                candidate_start = float(peak) - float(offset) * ppm
                predicted = np.array(
                    [candidate_start + expected * ppm for expected in expected_offsets],
                    dtype=float,
                )

                # 予測正時線と検出候補を近い順に一対一対応させる。
                pairs: list[tuple[float, int, int, float]] = []
                for pred_index, predicted_x in enumerate(predicted):
                    for peak_index, detected_x in enumerate(peaks):
                        distance = abs(float(detected_x) - float(predicted_x))
                        if distance <= tolerance:
                            pairs.append(
                                (distance, pred_index, peak_index, float(detected_x))
                            )

                pairs.sort(key=lambda item: item[0])
                used_pred: set[int] = set()
                used_peak: set[int] = set()
                matches: list[tuple[int, float]] = []
                residual_sum = 0.0

                for distance, pred_index, peak_index, detected_x in pairs:
                    if pred_index in used_pred or peak_index in used_peak:
                        continue
                    used_pred.add(pred_index)
                    used_peak.add(peak_index)
                    matches.append((expected_offsets[pred_index], detected_x))
                    residual_sum += distance

                if len(matches) < 2:
                    continue

                candidate_end = candidate_start + duration_minutes * ppm
                boundary_penalty = (
                    abs(candidate_start - rough_left)
                    + abs(candidate_end - rough_right)
                )
                spacing_penalty = abs(ppm - rough_ppm) * 60.0

                # 対応本数を最優先。次に残差、粗い枠との整合性。
                score = (
                    len(matches) * 100000.0
                    - residual_sum * 100.0
                    - boundary_penalty * 2.0
                    - spacing_penalty * 10.0
                )

                if candidate_start < -40 or candidate_end > image_width + 40:
                    score -= 100000.0

                if best is None or score > float(best["score"]):
                    best = {
                        "score": score,
                        "matches": matches,
                        "start": candidate_start,
                        "ppm": ppm,
                    }

    if best is None:
        raise RuntimeError("縦の時刻目盛りから時間軸を較正できませんでした。")

    matches = list(best["matches"])  # type: ignore[arg-type]

    # 対応した正時線だけで x = start + offset * ppm を最小二乗再推定。
    offsets = np.array([pair[0] for pair in matches], dtype=float)
    detected = np.array([pair[1] for pair in matches], dtype=float)
    design = np.column_stack([np.ones(len(offsets)), offsets])
    start_x, pixels_per_minute = np.linalg.lstsq(
        design, detected, rcond=None
    )[0]
    start_x = float(start_x)
    pixels_per_minute = float(pixels_per_minute)
    end_x = start_x + duration_minutes * pixels_per_minute

    predicted_matches = start_x + offsets * pixels_per_minute
    residuals = np.abs(predicted_matches - detected)
    hour_spacing = pixels_per_minute * 60.0
    allowed_residual = max(8.0, hour_spacing * 0.04)

    if float(np.max(residuals)) > allowed_residual:
        raise RuntimeError(
            "正時線の対応後も残差が大きく、安全に時間軸を確定できませんでした。"
        )

    # 粗いグラフ幅から極端に外れるモデルは拒否。
    allowed_boundary = max(50.0, rough_hour_spacing * 0.22)
    if (
        start_x < rough_left - allowed_boundary
        or end_x > rough_right + allowed_boundary
    ):
        raise RuntimeError(
            "推定した時間軸がグラフ領域から大きく外れています。"
        )

    matched_peak_values = {int(round(pair[1])) for pair in matches}
    outliers = [peak for peak in peaks if peak not in matched_peak_values]

    return start_x, end_x, pixels_per_minute, matches, outliers



def _find_colored_plot_bounds(
    arr: np.ndarray,
    graph_top: int,
    graph_bottom: int,
    rough_left: int,
    rough_right: int,
    tolerance: int = 50,
) -> tuple[int, int]:
    """
    局ごとに最長の色線区間を取り、その左右端の中央値を使う。

    6局の単純和集合だと、NHKカーソルや円の縁が10～20px外へ張り出す。
    中央値なら、一局だけ欠損・張り出しがあっても時間軸端を安定して推定できる。
    """
    x0 = max(0, rough_left - 120)
    x1 = min(arr.shape[1] - 1, rough_right + 120)
    roi = arr[graph_top:graph_bottom + 1, x0:x1 + 1]

    left_edges: list[int] = []
    right_edges: list[int] = []

    for station, color in STATION_COLORS.items():
        mask = _station_color_mask(
            roi,
            station,
            color,
            tolerance,
        )
        columns = np.where(mask.sum(axis=0) > 0)[0]
        if len(columns) < 10:
            continue

        groups = _group_consecutive(columns, max_gap=8)
        longest = max(
            groups,
            key=lambda group: group[-1] - group[0] + 1,
        )

        if longest[-1] - longest[0] + 1 < arr.shape[1] * 0.25:
            continue

        left_edges.append(x0 + longest[0])
        right_edges.append(x0 + longest[-1])

    if len(left_edges) < 3:
        raise RuntimeError("色線から時間軸範囲を推定できませんでした。")

    left = int(round(float(np.median(left_edges))))
    right = int(round(float(np.median(right_edges))))

    if right - left < arr.shape[1] * 0.45:
        raise RuntimeError("色線から得た時間軸範囲が狭すぎます。")

    return left, right


def _calibrate_time_axis(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
    end_time: datetime,
    duration_minutes: int,
) -> tuple[float, float, float, str, int]:
    """
    時間軸は次の順で求める。

    1. 複数の正時破線をRANSAC風に照合
    2. 失敗時は6局色線の実データ横範囲へフォールバック

    「時刻線が足りない」「余計な線が混ざった」だけでは解析を止めない。
    """
    peaks = _find_vertical_hour_lines(
        arr,
        grid[0],
        grid[-1],
        rough_left,
        rough_right,
    )

    start_time = end_time - timedelta(minutes=duration_minutes)
    first_hour_offset = (-start_time.minute) % 60
    expected_offsets = list(
        range(first_hour_offset, duration_minutes + 1, 60)
    )

    if len(peaks) >= 2 and len(expected_offsets) >= 2:
        try:
            (
                start_x,
                end_x,
                pixels_per_minute,
                matches,
                _,
            ) = _match_time_axis_model(
                peaks=peaks,
                expected_offsets=expected_offsets,
                rough_left=rough_left,
                rough_right=rough_right,
                duration_minutes=duration_minutes,
                image_width=arr.shape[1],
            )
            return (
                float(start_x),
                float(end_x),
                float(pixels_per_minute),
                "hour-lines",
                int(len(matches)),
            )
        except Exception:
            # 下の色線範囲フォールバックへ進む。
            pass

    color_left, color_right = _find_colored_plot_bounds(
        arr,
        grid[0],
        grid[-1],
        rough_left,
        rough_right,
        tolerance=50,
    )

    start_x = float(color_left)
    end_x = float(color_right)
    pixels_per_minute = (end_x - start_x) / float(duration_minutes)

    if pixels_per_minute <= 0:
        raise RuntimeError("時間軸の横幅が不正です。")

    return start_x, end_x, pixels_per_minute, "color-range-fallback", 0



def _find_cursor_x(
    arr: np.ndarray,
    grid: list[int],
    rough_left: int,
    rough_right: int,
) -> int | None:
    """
    選択時刻の実線カーソルを探す。

    正時破線より縦方向の灰色率が高く、通常1～3pxの細い実線になる。
    右端UIの太い灰色帯は幅で除外する。
    """
    try:
        color_left, color_right = _find_colored_plot_bounds(
            arr,
            grid[0],
            grid[-1],
            rough_left,
            rough_right,
            tolerance=50,
        )
    except Exception:
        color_left, color_right = rough_left, rough_right

    roi = arr[grid[0]:grid[-1] + 1]
    mask = _neutral_gray_mask(
        roi,
        max_spread=35,
        min_brightness=150,
        max_brightness=253,
    )
    scores = mask.mean(axis=0)

    x_axis = np.arange(arr.shape[1])
    candidates = np.where(
        (scores >= 0.82)
        & (x_axis >= color_left - 10)
        & (x_axis <= color_right + 10)
    )[0]
    groups = _group_consecutive(candidates, max_gap=2)

    thin_groups = [group for group in groups if 1 <= len(group) <= 8]
    if not thin_groups:
        return None

    chosen = max(
        thin_groups,
        key=lambda group: float(np.max(scores[group])),
    )
    return int(round(float(np.mean(chosen))))


def _find_cursor_station_y(
    arr: np.ndarray,
    cursor_x: int,
    graph_top: int,
    graph_bottom: int,
    target_rgb: Tuple[int, int, int],
    tolerance: int,
) -> float | None:
    """
    カーソル上の局色円から中心Yを求める。
    特にNHKはカーソル線で通常追跡が隠れるため、表示時点の補助観測に使う。
    """
    window = 14
    x0 = max(0, cursor_x - window)
    x1 = min(arr.shape[1] - 1, cursor_x + window)

    roi = arr[
        graph_top:graph_bottom + 1,
        x0:x1 + 1,
    ].astype(np.int16)
    target = np.array(target_rgb, dtype=np.int16)
    color_mask = np.max(np.abs(roi - target), axis=2) <= tolerance
    row_counts = color_mask.sum(axis=1)

    maximum = int(row_counts.max())
    if maximum < 2:
        return None

    threshold = max(2, int(np.ceil(maximum * 0.45)))
    rows = np.where(row_counts >= threshold)[0]
    groups = _group_consecutive(rows, max_gap=2)
    if not groups:
        return None

    chosen = max(
        groups,
        key=lambda group: int(row_counts[group].sum()),
    )
    y_values = np.array(chosen, dtype=float)
    weights = row_counts[chosen].astype(float)

    return float(np.average(y_values, weights=weights) + graph_top)


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
        # JPEG圧縮されたNHK灰色も拾う一方、明るいグリッド線は除外。
        spread = work.max(axis=2) - work.min(axis=2)
        brightness = work.mean(axis=2)
        return (
            (distance <= 55)
            & (brightness < 215)
            & (spread <= 25)
        )

    return distance <= tolerance


def _trace_station(
    arr: np.ndarray,
    station: str,
    color: Tuple[int, int, int],
    trace_left: int,
    trace_right: int,
    graph_top: int,
    graph_bottom: int,
    pixels_per_percent: float,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """
    時間軸の中央付近から左右へ追跡する。

    左端の選択カーソル丸を開始点にして別の灰色へ張り付く事故を防ぐ。
    """
    x0 = max(0, int(trace_left))
    x1 = min(arr.shape[1] - 1, int(trace_right))

    roi = arr[
        graph_top:graph_bottom + 1,
        x0:x1 + 1,
    ]
    mask = _station_color_mask(roi, station, color, tolerance)

    column_counts = mask.sum(axis=0)

    # NHKの灰色カーソル丸は周辺列まで灰色が多いため、専用の低い上限を使う。
    # これでカーソル円周を折れ線として読む事故を防ぐ。
    vertical_limit = (
        24
        if station == "NHK総合"
        else max(80, int(mask.shape[0] * 0.15))
    )
    mask[:, column_counts > vertical_limit] = False

    xs = np.arange(x0, x1 + 1)
    ys = np.full(len(xs), np.nan)
    center = len(xs) // 2

    anchor_index: int | None = None
    anchor_y: float | None = None

    # 中央から外へ探し、候補が一つだけの安定した列を開始点にする。
    for radius in range(len(xs)):
        test_indices = [center - radius, center + radius]
        for index in test_indices:
            if index < 0 or index >= len(xs):
                continue

            positions = np.where(mask[:, index])[0]
            if len(positions) == 0:
                continue

            groups = _group_consecutive(positions, max_gap=1)
            strong = [group for group in groups if len(group) >= 2]

            if len(strong) == 1:
                anchor_index = index
                anchor_y = float(np.median(strong[0])) + graph_top
                break

        if anchor_index is not None:
            break

    if anchor_index is None or anchor_y is None:
        # 候補が一つだけの列がない場合でも、中央に最も近い色成分から開始する。
        fallback_candidates: list[tuple[int, float, int]] = []
        for index in range(len(xs)):
            positions = np.where(mask[:, index])[0]
            if len(positions) == 0:
                continue
            groups = _group_consecutive(positions, max_gap=1)
            for group in groups:
                fallback_candidates.append(
                    (
                        index,
                        float(np.median(group)) + graph_top,
                        len(group),
                    )
                )

        if not fallback_candidates:
            raise RuntimeError(f"{station}の色線を一画素も検出できませんでした。")

        anchor_index, anchor_y, _ = min(
            fallback_candidates,
            key=lambda item:
                abs(item[0] - center) - min(item[2], 20) * 0.5,
        )

    ys[anchor_index] = anchor_y
    rejected_jumps = 0
    max_single_column_jump = max(70.0, pixels_per_percent * 1.5)

    for direction in (1, -1):
        previous_y = anchor_y
        index = anchor_index + direction

        while 0 <= index < len(xs):
            positions = np.where(mask[:, index])[0]

            if len(positions):
                groups = _group_consecutive(positions, max_gap=1)
                candidates = [
                    (
                        float(np.median(group)) + graph_top,
                        len(group),
                    )
                    for group in groups
                ]

                chosen = min(
                    candidates,
                    key=lambda candidate:
                        abs(candidate[0] - previous_y)
                        - min(candidate[1], 20) * 0.1,
                )

                if abs(chosen[0] - previous_y) <= max_single_column_jump:
                    ys[index] = chosen[0]
                    previous_y = chosen[0]
                else:
                    rejected_jumps += 1

            index += direction

    good = ~np.isnan(ys)
    coverage = float(good.mean())

    missing_groups = _group_consecutive(np.where(~good)[0], max_gap=1)
    max_gap_pixels = max((len(group) for group in missing_groups), default=0)

    if good.sum() < 2:
        raise RuntimeError(f"{station}の色線を十分に検出できませんでした。")

    interpolated = np.interp(xs, xs[good], ys[good])

    # 以前のように86px欠けただけで全体を拒否しない。
    # 欠損量を品質情報としてCSV・画面へ返す。
    if coverage >= 0.90 and max_gap_pixels <= 90:
        quality = "A"
    elif coverage >= 0.75 and max_gap_pixels <= 180:
        quality = "B"
    else:
        quality = "C"

    diagnostics: dict[str, float | int | str] = {
        "coverage": coverage,
        "max_gap_pixels": int(max_gap_pixels),
        "rejected_jumps": int(rejected_jumps),
        "quality": quality,
    }
    return xs, interpolated, diagnostics

def analyze_image(
    arr: np.ndarray,
    end_time: datetime,
    duration_minutes: int = 180,
    tolerance: int = 42,
) -> tuple[pd.DataFrame, Calibration, dict[str, dict[str, float | int | str]]]:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB画像を指定してください。")

    thin_lines = _find_thin_horizontal_lines(arr)
    grid, grid_gap = _select_regular_grid(thin_lines, arr.shape[0])

    graph_top = int(grid[0])
    y_zero = int(grid[-1])
    pixels_per_percent = grid_gap / 2.0

    rough_left, rough_right = _find_rough_graph_x_bounds(arr, grid)
    (
        start_x,
        end_x,
        pixels_per_minute,
        time_axis_method,
        time_line_inliers,
    ) = _calibrate_time_axis(
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
        time_axis_method=time_axis_method,
        time_line_inliers=time_line_inliers,
    )

    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, float | int | str]] = {}

    for station, color in STATION_COLORS.items():
        xs, ys, station_diagnostics = _trace_station(
            arr,
            station,
            color,
            max(0, int(np.floor(start_x)) - 4),
            min(arr.shape[1] - 1, int(np.ceil(end_x)) + 4),
            graph_top,
            y_zero,
            pixels_per_percent,
            tolerance,
        )
        gap_minutes = (
            float(station_diagnostics["max_gap_pixels"])
            / pixels_per_minute
        )
        station_diagnostics["max_gap_minutes"] = round(gap_minutes, 2)

        coverage = float(station_diagnostics["coverage"])
        if coverage >= 0.90 and gap_minutes <= 5.0:
            station_diagnostics["quality"] = "A"
        elif coverage >= 0.75 and gap_minutes <= 15.0:
            station_diagnostics["quality"] = "B"
        else:
            station_diagnostics["quality"] = "C"

        traces[station] = (xs, ys)
        diagnostics[station] = station_diagnostics

    # 選択カーソルでNHK線が隠れた場合、灰色円の中心を補助観測として戻す。
    cursor_x = _find_cursor_x(
        arr,
        grid,
        rough_left,
        rough_right,
    )
    if cursor_x is not None and "NHK総合" in traces:
        cursor_y = _find_cursor_station_y(
            arr,
            cursor_x,
            graph_top,
            y_zero,
            STATION_COLORS["NHK総合"],
            tolerance=20,
        )
        if cursor_y is not None:
            nhk_xs, nhk_ys = traces["NHK総合"]
            nearest = int(np.argmin(np.abs(nhk_xs - cursor_x)))
            left = max(0, nearest - 8)
            right = min(len(nhk_ys), nearest + 9)
            nhk_ys[left:right] = cursor_y
            traces["NHK総合"] = (nhk_xs, nhk_ys)
            diagnostics["NHK総合"]["cursor_anchor_used"] = 1
        else:
            diagnostics["NHK総合"]["cursor_anchor_used"] = 0

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
