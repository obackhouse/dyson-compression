"""Base class for solvers.
"""

import numpy as np
import logging
import sys

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


class Solver:
    def __init__(self, *inp, **kwargs):
        self.log = kwargs.pop("log", log)
        self.inp = inp
        self.res = None

        for key, val in kwargs.items():
            assert key in self.__dict__
            setattr(self, key, val)

        self._preamble()

    def kernel(self):
        raise NotImplementedError

    def preamble(self):
        raise NotImplementedError

    def _preamble(self):
        self.log.info(self.__class__.__name__)
        self.log.info("*" * len(self.__class__.__name__))

        self.preamble()

    @property
    def norb(self):
        return NotImplemented

    @property
    def dtype(self):
        return NotImplemented
