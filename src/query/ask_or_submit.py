from __future__ import annotations


def should_ask(current_map: float, expected_next_map: float, gamma: float) -> bool:
    # Ask if discounted value of information beats current confidence.
    return gamma * expected_next_map > current_map
