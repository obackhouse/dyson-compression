"""Utility functions.
"""

import numpy as np

def format_value(v, prec=12):
    """Format a value to a given precision.
    """

    if not np.iscomplexobj(v):
        return "%.*f" % (prec, v)
    else:
        r, i = v.real, v.imag
        i, op = np.abs(i), "+" if i >= 0 else "-"
        return "%.*f %s %.*fj" % (prec, r, op, prec, i)
