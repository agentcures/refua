from abc import ABC, abstractmethod

import numpy as np
from boltzgen.data.data import Tokenized


class Selector(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def select(
        self,
        data: Tokenized,
        random: np.random.Generator,
        chain_id: int | None = None,
        interface_id: int | None = None,
    ) -> Tokenized:
        """Select which part of tokenized should be designed

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        random : np.random.Generator
            The random state for reproducibility.
        
        Returns
        -------
        Tokenized
            The data with updated design entry.

        """
        raise NotImplementedError
