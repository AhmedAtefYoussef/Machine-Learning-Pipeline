"""P1 -> P2 weight lift."""
import numpy as np


def lift_weights(w_small, names_small, names_big):
    """Place the P1 weights in their slots of the bigger design, zeros elsewhere.

    w_small (p_small,), names_small / names_big are column-name lists.
    Returns w_big (p_big,) with w_big[names_big.index(name)] = w_small[i].
    Because the extra columns start at weight 0, the initial P2 loss equals the final P1 loss.
    """
    slot = {name: j for j, name in enumerate(names_big)}
    missing = [name for name in names_small if name not in slot]
    assert not missing, f"names missing from the big design: {missing}"
    w_big = np.zeros(len(names_big), dtype=np.float64)
    for i, name in enumerate(names_small):
        w_big[slot[name]] = w_small[i]
    return w_big
