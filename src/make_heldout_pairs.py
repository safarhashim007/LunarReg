import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import array_bounds


# =========================================================
# INPUTS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

TMC_PATH = (
    ROOT
    / "data/chandrayaan/tmc_20240623/"
      "ch2_tmc_ndn_20240623T0707183819_d_oth_d18.tif"
)

LRO_PATH = (
    ROOT
    / "data/reference/"
      "NAC_ROI_COPERNICLOA_E098N3401_20M.TIF"
)

OUT_ROOT = ROOT / "data/test_pairs"


# Existing tuned/test areas — exclude them
EXCLUDED_AOIS = [
    # pair_001
    {
        "lon_min": 339.753,
        "lon_max": 340.087,
        "lat_min": 9.455,
        "lat_max": 9.785,
    },

    # pair_002
    {
        "lon_min": 340.120,
        "lon_max": 340.420,
        "lat_min": 9.455,
        "lat_max": 9.785,
    },
]


# Approx same physical size as pair_002
AOI_WIDTH_DEG = 0.30
AOI_HEIGHT_DEG = 0.33

N_PAIRS = 4

PAIR_START = 3

MIN_TMC_VALID = 0.95
MIN_LRO_VALID = 0.95

# Candidate spacing
STEP_LON = 0.10
STEP_LAT = 0.11


# Lunar projection used by LRO mosaic
MOON_RADIUS = 1737400.0
STANDARD_PARALLEL_DEG = 10.0


# =========================================================
# HELPERS
# =========================================================

def lonlat_to_lro_xy(lon_deg, lat_deg):
    """
    Preserve the source's 0–360 longitude convention.
    """

    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)

    std = math.radians(STANDARD_PARALLEL_DEG)

    x = (
        MOON_RADIUS
        * lon_rad
        * math.cos(std)
    )

    y = (
        MOON_RADIUS
        * lat_rad
    )

    return x, y


def lro_xy_to_lonlat(x, y):
    std = math.radians(STANDARD_PARALLEL_DEG)

    lon = math.degrees(
        x
        /
        (
            MOON_RADIUS
            * math.cos(std)
        )
    )

    lat = math.degrees(
        y / MOON_RADIUS
    )

    return lon, lat


def clamp_window(window, width, height):
    col0 = max(0, int(math.floor(window.col_off)))
    row0 = max(0, int(math.floor(window.row_off)))

    col1 = min(
        width,
        int(math.ceil(window.col_off + window.width))
    )

    row1 = min(
        height,
        int(math.ceil(window.row_off + window.height))
    )

    if col1 <= col0 or row1 <= row0:
        return None

    return Window(
        col0,
        row0,
        col1 - col0,
        row1 - row0
    )


def overlaps(a, b, margin=0.01):
    return not (
        a["lon_max"] + margin <= b["lon_min"]
        or
        a["lon_min"] - margin >= b["lon_max"]
        or
        a["lat_max"] + margin <= b["lat_min"]
        or
        a["lat_min"] - margin >= b["lat_max"]
    )


def valid_fraction(array, nodata=None):
    arr = array.astype(np.float32)

    mask = np.isfinite(arr)

    if nodata is not None:
        mask &= arr != nodata

    # Lunar products here use zero extensively for empty/no-data.
    mask &= arr > 0

    return float(mask.mean())


def preview_valid_fraction(ds, window):
    """
    Read a small version so candidate search stays fast.
    """

    data = ds.read(
        1,
        window=window,
        out_shape=(128, 128),
        resampling=rasterio.enums.Resampling.nearest,
    )

    return valid_fraction(
        data,
        ds.nodata
    )


def normalize_png(arr, nodata=None):
    data = arr.astype(np.float32)

    valid = np.isfinite(data)

    if nodata is not None:
        valid &= data != nodata

    valid &= data > 0

    out = np.zeros(
        data.shape,
        dtype=np.uint8
    )

    if valid.sum() < 10:
        return out

    values = data[valid]

    lo = float(
        np.percentile(values, 1.0)
    )

    hi = float(
        np.percentile(values, 99.0)
    )

    if hi <= lo:
        hi = lo + 1.0

    scaled = (
        (data - lo)
        /
        (hi - lo)
        * 255.0
    )

    scaled = np.clip(
        scaled,
        0,
        255
    )

    out[valid] = scaled[valid].astype(
        np.uint8
    )

    return out


def write_crop(ds, window, output_path):
    data = ds.read(
        1,
        window=window
    )

    transform = ds.window_transform(
        window
    )

    profile = ds.profile.copy()

    profile.update(
        {
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
            "count": 1,
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:
        dst.write(
            data,
            1
        )

    return data


# =========================================================
# OPEN SOURCE PRODUCTS
# =========================================================

print("===== HELD-OUT PAIR GENERATOR =====")
print()

print("TMC:")
print(TMC_PATH)

print()
print("LRO:")
print(LRO_PATH)

if not TMC_PATH.exists():
    raise RuntimeError(
        f"Missing {TMC_PATH}"
    )

if not LRO_PATH.exists():
    raise RuntimeError(
        f"Missing {LRO_PATH}"
    )


with rasterio.open(TMC_PATH) as tmc, rasterio.open(LRO_PATH) as lro:

    print()
    print("TMC size:", tmc.width, "x", tmc.height)
    print("TMC bounds:", tmc.bounds)

    print()
    print("LRO size:", lro.width, "x", lro.height)
    print("LRO bounds:", lro.bounds)


    # =====================================================
    # LRO GEOGRAPHIC EXTENT
    # =====================================================

    lro_lon_1, lro_lat_1 = lro_xy_to_lonlat(
        lro.bounds.left,
        lro.bounds.bottom,
    )

    lro_lon_2, lro_lat_2 = lro_xy_to_lonlat(
        lro.bounds.right,
        lro.bounds.top,
    )

    lro_lon_min = min(
        lro_lon_1,
        lro_lon_2
    )

    lro_lon_max = max(
        lro_lon_1,
        lro_lon_2
    )

    lro_lat_min = min(
        lro_lat_1,
        lro_lat_2
    )

    lro_lat_max = max(
        lro_lat_1,
        lro_lat_2
    )


    print()
    print("Approx LRO geographic extent:")

    print(
        "Lon:",
        round(lro_lon_min, 6),
        "to",
        round(lro_lon_max, 6),
    )

    print(
        "Lat:",
        round(lro_lat_min, 6),
        "to",
        round(lro_lat_max, 6),
    )


    # =====================================================
    # INTERSECTION WITH TMC
    # =====================================================

    tmc_lon_min = min(
        tmc.bounds.left,
        tmc.bounds.right
    )

    tmc_lon_max = max(
        tmc.bounds.left,
        tmc.bounds.right
    )

    tmc_lat_min = min(
        tmc.bounds.bottom,
        tmc.bounds.top
    )

    tmc_lat_max = max(
        tmc.bounds.bottom,
        tmc.bounds.top
    )


    overlap_lon_min = max(
        tmc_lon_min,
        lro_lon_min
    )

    overlap_lon_max = min(
        tmc_lon_max,
        lro_lon_max
    )

    overlap_lat_min = max(
        tmc_lat_min,
        lro_lat_min
    )

    overlap_lat_max = min(
        tmc_lat_max,
        lro_lat_max
    )


    print()
    print("Common geographic overlap:")

    print(
        "Lon:",
        round(overlap_lon_min, 6),
        "to",
        round(overlap_lon_max, 6),
    )

    print(
        "Lat:",
        round(overlap_lat_min, 6),
        "to",
        round(overlap_lat_max, 6),
    )


    if (
        overlap_lon_max - overlap_lon_min
        < AOI_WIDTH_DEG
        or
        overlap_lat_max - overlap_lat_min
        < AOI_HEIGHT_DEG
    ):
        raise RuntimeError(
            "Common overlap is too small for held-out pairs."
        )


    # =====================================================
    # SEARCH CANDIDATE AOIs
    # =====================================================

    candidates = []

    lat = overlap_lat_min

    while (
        lat + AOI_HEIGHT_DEG
        <= overlap_lat_max
    ):

        lon = overlap_lon_min

        while (
            lon + AOI_WIDTH_DEG
            <= overlap_lon_max
        ):

            aoi = {
                "lon_min": lon,
                "lon_max": lon + AOI_WIDTH_DEG,
                "lat_min": lat,
                "lat_max": lat + AOI_HEIGHT_DEG,
            }


            # ---------------------------------------------
            # Exclude tuned areas
            # ---------------------------------------------

            if any(
                overlaps(
                    aoi,
                    excluded
                )
                for excluded in EXCLUDED_AOIS
            ):
                lon += STEP_LON
                continue


            # ---------------------------------------------
            # TMC window
            # ---------------------------------------------

            tmc_window = from_bounds(
                aoi["lon_min"],
                aoi["lat_min"],
                aoi["lon_max"],
                aoi["lat_max"],
                transform=tmc.transform,
            )

            tmc_window = clamp_window(
                tmc_window,
                tmc.width,
                tmc.height
            )

            if tmc_window is None:
                lon += STEP_LON
                continue


            # ---------------------------------------------
            # LRO window
            # ---------------------------------------------

            x_min, y_min = lonlat_to_lro_xy(
                aoi["lon_min"],
                aoi["lat_min"],
            )

            x_max, y_max = lonlat_to_lro_xy(
                aoi["lon_max"],
                aoi["lat_max"],
            )


            lro_window = from_bounds(
                min(x_min, x_max),
                min(y_min, y_max),
                max(x_min, x_max),
                max(y_min, y_max),
                transform=lro.transform,
            )

            lro_window = clamp_window(
                lro_window,
                lro.width,
                lro.height
            )

            if lro_window is None:
                lon += STEP_LON
                continue


            tmc_valid = preview_valid_fraction(
                tmc,
                tmc_window
            )

            lro_valid = preview_valid_fraction(
                lro,
                lro_window
            )


            if (
                tmc_valid >= MIN_TMC_VALID
                and
                lro_valid >= MIN_LRO_VALID
            ):

                score = (
                    tmc_valid
                    +
                    lro_valid
                )

                candidates.append(
                    {
                        **aoi,
                        "tmc_valid": tmc_valid,
                        "lro_valid": lro_valid,
                        "score": score,
                    }
                )


            lon += STEP_LON

        lat += STEP_LAT


# =========================================================
# SELECT SPATIALLY DISTINCT AOIs
# =========================================================

print()
print(
    "Valid candidate AOIs found:",
    len(candidates)
)


candidates.sort(
    key=lambda x: x["score"],
    reverse=True
)


selected = []


for candidate in candidates:

    if any(
        overlaps(
            candidate,
            chosen,
            margin=0.03
        )
        for chosen in selected
    ):
        continue

    selected.append(
        candidate
    )

    if len(selected) == N_PAIRS:
        break


if len(selected) < N_PAIRS:
    print()
    print(
        "WARNING:",
        f"Only {len(selected)} spatially distinct "
        f"high-validity AOIs were available."
    )


# =========================================================
# CREATE PAIRS
# =========================================================

manifest = []


with rasterio.open(TMC_PATH) as tmc, rasterio.open(LRO_PATH) as lro:

    for offset, aoi in enumerate(selected):

        pair_number = (
            PAIR_START
            +
            offset
        )

        pair_name = (
            f"pair_{pair_number:03d}"
        )

        pair_dir = (
            OUT_ROOT
            /
            pair_name
        )

        pair_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        print()
        print(
            "======================================"
        )

        print(
            "Creating",
            pair_name
        )

        print(
            "AOI:",
            f"lon {aoi['lon_min']:.6f}"
            f"–{aoi['lon_max']:.6f},",
            f"lat {aoi['lat_min']:.6f}"
            f"–{aoi['lat_max']:.6f}"
        )


        # ---------------------------------------------
        # TMC crop
        # ---------------------------------------------

        tmc_window = from_bounds(
            aoi["lon_min"],
            aoi["lat_min"],
            aoi["lon_max"],
            aoi["lat_max"],
            transform=tmc.transform,
        )

        tmc_window = clamp_window(
            tmc_window,
            tmc.width,
            tmc.height
        )


        ch_tif = (
            pair_dir
            /
            "chandrayaan.tif"
        )

        ch_arr = write_crop(
            tmc,
            tmc_window,
            ch_tif
        )


        # ---------------------------------------------
        # LRO crop
        # ---------------------------------------------

        x_min, y_min = lonlat_to_lro_xy(
            aoi["lon_min"],
            aoi["lat_min"]
        )

        x_max, y_max = lonlat_to_lro_xy(
            aoi["lon_max"],
            aoi["lat_max"]
        )


        lro_window = from_bounds(
            min(x_min, x_max),
            min(y_min, y_max),
            max(x_min, x_max),
            max(y_min, y_max),
            transform=lro.transform,
        )

        lro_window = clamp_window(
            lro_window,
            lro.width,
            lro.height
        )


        lro_tif = (
            pair_dir
            /
            "lro.tif"
        )

        lro_arr = write_crop(
            lro,
            lro_window,
            lro_tif
        )


        # ---------------------------------------------
        # PNGs
        # ---------------------------------------------

        ch_png_arr = normalize_png(
            ch_arr,
            tmc.nodata
        )

        lro_png_arr = normalize_png(
            lro_arr,
            lro.nodata
        )


        cv2.imwrite(
            str(
                pair_dir
                /
                "chandrayaan.png"
            ),
            ch_png_arr
        )

        cv2.imwrite(
            str(
                pair_dir
                /
                "lro.png"
            ),
            lro_png_arr
        )


        # ---------------------------------------------
        # Side-by-side preview
        # ---------------------------------------------

        preview_height = 500


        ch_scale = (
            preview_height
            /
            ch_png_arr.shape[0]
        )

        lro_scale = (
            preview_height
            /
            lro_png_arr.shape[0]
        )


        ch_preview = cv2.resize(
            ch_png_arr,
            (
                max(
                    1,
                    int(
                        ch_png_arr.shape[1]
                        *
                        ch_scale
                    )
                ),
                preview_height
            )
        )


        lro_preview = cv2.resize(
            lro_png_arr,
            (
                max(
                    1,
                    int(
                        lro_png_arr.shape[1]
                        *
                        lro_scale
                    )
                ),
                preview_height
            )
        )


        side = np.hstack(
            [
                ch_preview,
                lro_preview
            ]
        )


        cv2.imwrite(
            str(
                pair_dir
                /
                "side_by_side.png"
            ),
            side
        )


        # ---------------------------------------------
        # Exact valid fractions
        # ---------------------------------------------

        tmc_valid = valid_fraction(
            ch_arr,
            tmc.nodata
        )

        lro_valid = valid_fraction(
            lro_arr,
            lro.nodata
        )


        metadata = {
            "pair": pair_name,

            "source_chandrayaan":
                str(
                    TMC_PATH.relative_to(ROOT)
                ),

            "source_reference":
                str(
                    LRO_PATH.relative_to(ROOT)
                ),

            "aoi": {
                "lon_min": aoi["lon_min"],
                "lon_max": aoi["lon_max"],
                "lat_min": aoi["lat_min"],
                "lat_max": aoi["lat_max"],
            },

            "chandrayaan_shape": [
                int(ch_arr.shape[0]),
                int(ch_arr.shape[1]),
            ],

            "lro_shape": [
                int(lro_arr.shape[0]),
                int(lro_arr.shape[1]),
            ],

            "chandrayaan_valid_fraction":
                tmc_valid,

            "lro_valid_fraction":
                lro_valid,

            "held_out": True,

            "selection_note":
                (
                    "Selected from shared geographic "
                    "coverage using source metadata and "
                    "valid-pixel fraction only. GeoTIFF "
                    "correspondence truth was not used "
                    "for registration or ranking."
                ),
        }


        with open(
            pair_dir
            /
            "metadata.json",
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2
            )


        manifest.append(
            metadata
        )


        print(
            "Chandrayaan:",
            ch_arr.shape,
            f"{tmc_valid * 100:.2f}% valid"
        )

        print(
            "LRO        :",
            lro_arr.shape,
            f"{lro_valid * 100:.2f}% valid"
        )


# =========================================================
# MANIFEST
# =========================================================

manifest_path = (
    OUT_ROOT
    /
    "heldout_manifest.json"
)


with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print()
print(
    "======================================"
)

print(
    "HELD-OUT PAIRS CREATED:",
    len(manifest)
)

print(
    "Manifest:",
    manifest_path
)

for item in manifest:

    print(
        item["pair"],
        "TMC valid",
        f"{item['chandrayaan_valid_fraction'] * 100:.2f}%",
        "| LRO valid",
        f"{item['lro_valid_fraction'] * 100:.2f}%"
    )

print()
print("Done.")
