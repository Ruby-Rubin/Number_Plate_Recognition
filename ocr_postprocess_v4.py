from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OCR_RESULTS = (
    PROJECT_ROOT
    / "runs"
    / "ocr_test_v3"
    / "ocr_results.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "plate_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "ocr_test_v4"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "ocr_results_v4.csv"
)


# ============================================================
# INDIAN STATE / UT PREFIXES
# ============================================================

STATE_PREFIXES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG",
    "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB",
    "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB"
}


# ============================================================
# CHARACTER CONFUSIONS
# ============================================================

# When a character is expected to be a DIGIT.
LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7"
}


# When a character is expected to be a LETTER.
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
    "7": "T"
}


# ============================================================
# NORMALIZE
# ============================================================

def normalize(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


# ============================================================
# PARSE A KNOWN GROUND-TRUTH PLATE
# ============================================================

def parse_plate_structure(plate):

    plate = normalize(plate)

    if len(plate) < 6:
        return None

    prefix = plate[:2]

    if prefix not in STATE_PREFIXES:
        return None

    remainder = plate[2:]


    # --------------------------------------------------------
    # Find initial digit segment
    # --------------------------------------------------------

    district_match = re.match(
        r"\d{1,2}",
        remainder
    )


    if district_match:

        district = district_match.group()

    else:

        return None


    pos = len(district)


    # --------------------------------------------------------
    # Find letter series
    # --------------------------------------------------------

    series_match = re.match(
        r"[A-Z]{0,3}",
        remainder[pos:]
    )


    series = series_match.group()

    pos += len(series)


    # --------------------------------------------------------
    # Remaining characters should be final digits
    # --------------------------------------------------------

    final_number = remainder[pos:]


    if not final_number.isdigit():

        return None


    if not (1 <= len(final_number) <= 4):

        return None


    return (
        len(district),
        len(series),
        len(final_number)
    )


# ============================================================
# LEARN COMMON PLATE STRUCTURE FROM TRAINING DATA
# ============================================================

def learn_patterns(metadata):

    patterns = defaultdict(Counter)


    train_data = metadata[
        metadata["split"]
        .astype(str)
        .str.lower()
        == "train"
    ]


    for _, row in train_data.iterrows():

        plate = normalize(
            row["plate_text"]
        )

        parsed = parse_plate_structure(
            plate
        )

        if parsed is None:
            continue


        state = plate[:2]

        patterns[state][parsed] += 1


    return patterns


# ============================================================
# FIND STATE PREFIX INSIDE OCR TEXT
# ============================================================

def find_state_prefix(text):

    text = normalize(text)

    # Exact beginning
    if (
        len(text) >= 2
        and text[:2] in STATE_PREFIXES
    ):

        return (
            text[:2],
            text[2:]
        )


    # OCR occasionally inserts one or two characters
    # before the actual state code.

    for start in [1, 2]:

        if start + 2 > len(text):
            continue

        candidate = text[
            start:start + 2
        ]

        if candidate in STATE_PREFIXES:

            return (
                candidate,
                text[
                    start + 2:
                ]
            )


    return None, text


# ============================================================
# CHARACTER TYPE COMPATIBILITY
# ============================================================

def expected_cost(char, expected_type):

    if expected_type == "digit":

        if char.isdigit():
            return 0.0

        if char in LETTER_TO_DIGIT:
            return 0.4

        return 2.0


    if expected_type == "letter":

        if char.isalpha():
            return 0.0

        if char in DIGIT_TO_LETTER:
            return 0.4

        return 2.0


    return 0.0


# ============================================================
# CORRECT CHARACTER
# ============================================================

def correct_character(
    char,
    expected_type
):

    if expected_type == "digit":

        if char.isdigit():
            return char

        return LETTER_TO_DIGIT.get(
            char,
            char
        )


    if expected_type == "letter":

        if char.isalpha():
            return char

        return DIGIT_TO_LETTER.get(
            char,
            char
        )


    return char


# ============================================================
# BUILD EXPECTED CHARACTER TYPES
# ============================================================

def build_types(pattern):

    district_len, series_len, final_len = pattern

    types = []

    types.extend(
        ["digit"] * district_len
    )

    types.extend(
        ["letter"] * series_len
    )

    types.extend(
        ["digit"] * final_len
    )

    return types


# ============================================================
# CORRECT AGAINST A STRUCTURE
# ============================================================

def correct_against_pattern(
    remainder,
    pattern
):

    expected_types = build_types(
        pattern
    )

    target_length = len(
        expected_types
    )

    remainder = normalize(
        remainder
    )


    # --------------------------------------------------------
    # Same length
    # --------------------------------------------------------

    if len(remainder) == target_length:

        corrected = []
        cost = 0.0


        for char, expected_type in zip(
            remainder,
            expected_types
        ):

            cost += expected_cost(
                char,
                expected_type
            )

            corrected.append(
                correct_character(
                    char,
                    expected_type
                )
            )


        return (
            "".join(corrected),
            cost
        )


    # --------------------------------------------------------
    # One extra OCR character
    # --------------------------------------------------------

    if len(remainder) == target_length + 1:

        best_text = None
        best_cost = float("inf")


        for remove_index in range(
            len(remainder)
        ):

            candidate = (
                remainder[:remove_index]
                +
                remainder[
                    remove_index + 1:
                ]
            )


            corrected, cost = (
                correct_against_pattern(
                    candidate,
                    pattern
                )
            )


            if cost < best_cost:

                best_cost = cost

                best_text = corrected


        return (
            best_text,
            best_cost + 0.5
        )


    # --------------------------------------------------------
    # Too short / too long
    # --------------------------------------------------------

    return (
        remainder,
        5.0
        + abs(
            len(remainder)
            - target_length
        )
    )


# ============================================================
# CORRECT OCR TEXT
# ============================================================

def correct_plate(
    raw_text,
    patterns
):

    raw_text = normalize(
        raw_text
    )


    if not raw_text:

        return (
            "",
            0.0,
            "none"
        )


    state, remainder = (
        find_state_prefix(
            raw_text
        )
    )


    # No recognizable state prefix.
    if state is None:

        return (
            raw_text,
            0.0,
            "unknown-state"
        )


    # --------------------------------------------------------
    # Get learned patterns for this state
    # --------------------------------------------------------

    state_patterns = patterns.get(
        state,
        Counter()
    )


    if not state_patterns:

        return (
            state + remainder,
            0.0,
            "no-pattern"
        )


    candidates = []


    # --------------------------------------------------------
    # Try all learned structures
    # --------------------------------------------------------

    for pattern, frequency in (
        state_patterns.items()
    ):

        corrected, cost = (
            correct_against_pattern(
                remainder,
                pattern
            )
        )


        # Stronger preference for common patterns.
        frequency_bonus = (
            frequency ** 0.5
        )


        total_score = (
            cost
            - frequency_bonus
        )


        candidates.append(
            (
                total_score,
                state
                + corrected,
                pattern,
                frequency
            )
        )


    # --------------------------------------------------------
    # Select best
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[0]
    )


    best_score, best_text, pattern, frequency = (
        candidates[0]
    )


    return (
        best_text,
        best_score,
        str(pattern)
    )


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [
        [0] * cols
        for _ in range(rows)
    ]


    for i in range(rows):
        dp[i][0] = i


    for j in range(cols):
        dp[0][j] = j


    for i in range(1, rows):

        for j in range(1, cols):

            substitution_cost = (
                0
                if a[i - 1] == b[j - 1]
                else 1
            )


            dp[i][j] = min(

                dp[i - 1][j] + 1,

                dp[i][j - 1] + 1,

                dp[i - 1][j - 1]
                + substitution_cost
            )


    return dp[-1][-1]


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def character_similarity(
    predicted,
    actual
):

    if not actual:
        return 0.0


    distance = edit_distance(
        predicted,
        actual
    )


    return max(
        0.0,
        1.0
        - (
            distance
            / max(
                len(predicted),
                len(actual)
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("NUMBER PLATE OCR - V4 POST PROCESSOR")
    print("========================================")


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not OCR_RESULTS.exists():

        raise FileNotFoundError(
            f"OCR result file not found:\n"
            f"{OCR_RESULTS}"
        )


    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    results = pd.read_csv(
        OCR_RESULTS
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )


    print(
        "\nV3 results:",
        len(results)
    )


    print(
        "Metadata:",
        len(metadata)
    )


    # --------------------------------------------------------
    # LEARN TRAINING PATTERNS
    # --------------------------------------------------------

    print(
        "\nLearning plate structures "
        "from training split..."
    )


    patterns = learn_patterns(
        metadata
    )


    print(
        "States learned:",
        len(patterns)
    )


    # --------------------------------------------------------
    # PROCESS OCR RESULTS
    # --------------------------------------------------------

    corrected_results = []


    exact_matches_before = 0
    exact_matches_after = 0


    similarity_before = 0.0
    similarity_after = 0.0


    evaluated = 0


    for _, row in results.iterrows():

        raw_text = normalize(
            row["ocr_text"]
        )

        ground_truth = normalize(
            row["ground_truth"]
        )


        corrected, score, pattern = (
            correct_plate(
                raw_text,
                patterns
            )
        )


        before_similarity = (
            character_similarity(
                raw_text,
                ground_truth
            )
        )


        after_similarity = (
            character_similarity(
                corrected,
                ground_truth
            )
        )


        before_exact = (
            raw_text == ground_truth
        )


        after_exact = (
            corrected == ground_truth
        )


        if ground_truth:

            evaluated += 1

            similarity_before += (
                before_similarity
            )

            similarity_after += (
                after_similarity
            )


            if before_exact:

                exact_matches_before += 1


            if after_exact:

                exact_matches_after += 1


        corrected_results.append({

            "image":
                row["image"],

            "ocr_text_v3":
                raw_text,

            "ocr_text_v4":
                corrected,

            "ground_truth":
                ground_truth,

            "character_similarity_v3":
                round(
                    before_similarity,
                    4
                ),

            "character_similarity_v4":
                round(
                    after_similarity,
                    4
                ),

            "exact_match_v3":
                before_exact,

            "exact_match_v4":
                after_exact,

            "learned_pattern":
                pattern,

            "correction_score":
                round(
                    score,
                    4
                )
        })


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_df = pd.DataFrame(
        corrected_results
    )


    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print("\n========================================")
    print("V4 RESULTS")
    print("========================================")


    if evaluated > 0:

        exact_before = (
            exact_matches_before
            / evaluated
        )

        exact_after = (
            exact_matches_after
            / evaluated
        )

        sim_before = (
            similarity_before
            / evaluated
        )

        sim_after = (
            similarity_after
            / evaluated
        )


        print(
            "\nV3 exact-match:",
            f"{exact_before * 100:.2f}%"
        )


        print(
            "V4 exact-match:",
            f"{exact_after * 100:.2f}%"
        )


        print(
            "\nV3 character similarity:",
            f"{sim_before * 100:.2f}%"
        )


        print(
            "V4 character similarity:",
            f"{sim_after * 100:.2f}%"
        )


        print(
            "\nExact matches:",
            exact_matches_after,
            "/",
            evaluated
        )


    print(
        "\nSaved:",
        OUTPUT_CSV
    )


    # --------------------------------------------------------
    # SHOW CHANGED RESULTS
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "EXAMPLES OF CORRECTIONS"
    )

    print(
        "========================================"
    )


    changed = output_df[
        output_df["ocr_text_v3"]
        !=
        output_df["ocr_text_v4"]
    ]


    for _, row in changed.head(20).iterrows():

        print(
            f'{row["image"]}: '
            f'{row["ocr_text_v3"]} '
            f'-> '
            f'{row["ocr_text_v4"]} '
            f'| GT: '
            f'{row["ground_truth"]}'
        )