import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

"""Unit tests for biological signed rule propagation."""
from lbd.ranking.score_paths import calculate_net_sign, get_direction_fit

def test_golden_path_heart_failure_sign():
    # Dapagliflozin -[INHIBITS (-1)]-> SLC5A2 -[DECREASES (-1)]-> Natriuresis -[DECREASES (-1)]-> Preload -[CAUSES (+1)]-> Heart Failure
    edges = [
        {"sign": -1}, # INHIBITS
        {"sign": -1}, # DECREASES
        {"sign": -1}, # DECREASES
        {"sign": 1}   # CAUSES
    ]
    net_sign = calculate_net_sign(edges)
    assert net_sign == -1 # (-1)*(-1)*(-1)*(+1) = -1
    assert get_direction_fit(net_sign) == 1.5 # Therapeutic alignment

def test_golden_path_ckd_sign():
    # (-1) * (-1) * (+1) * (+1) * (-1) * (+1) = -1
    edges = [
        {"sign": -1},
        {"sign": -1},
        {"sign": 1},
        {"sign": 1},
        {"sign": -1},
        {"sign": 1}
    ]
    net_sign = calculate_net_sign(edges)
    assert net_sign == -1
    assert get_direction_fit(net_sign) == 1.5

def test_contradictory_path_sign():
    # If a path leads to net positive (+1), it is contradictory (aggravates disease)
    edges = [{"sign": -1}, {"sign": -1}]
    net_sign = calculate_net_sign(edges)
    assert net_sign == 1
    assert get_direction_fit(net_sign) == 0.0
