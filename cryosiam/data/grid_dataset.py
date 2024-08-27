import numpy as np
from typing import Dict, Sequence, Union
from monai.data.utils import iter_patch

from monai.utils import NumpyPadMode, ensure_tuple, look_up_option


class PatchIter:
    """
    Return a patch generator with predefined properties such as `patch_size`.
    Typically used with :py:class:`monai.data.GridPatchDataset`.
    """

    def __init__(self, patch_size: Sequence[int], start_pos: Sequence[int] = (),
                 overlap: Union[Sequence[float], float] = 0.0, mode: str = NumpyPadMode.WRAP, **pad_opts: Dict):
        """
        Args:
            patch_size: size of patches to generate slices for, 0/None selects whole dimension
            start_pos: starting position in the array, default is 0 for each dimension
            overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
                If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            pad_opts: other arguments for the `np.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.
        Note:
            The `patch_size` is the size of the
            patch to sample from the input arrays. It is assumed the arrays first dimension is the channel dimension which
            will be yielded in its entirety so this should not be specified in `patch_size`. For example, for an input 3D
            array with 1 channel of size (1, 20, 20, 20) a regular grid sampling of eight patches (1, 10, 10, 10) would be
            specified by a `patch_size` of (10, 10, 10).
        """
        self.patch_size = (None,) + tuple(patch_size)  # expand to have the channel dim
        self.start_pos = ensure_tuple(start_pos)
        self.overlap = overlap
        self.mode: NumpyPadMode = look_up_option(mode, NumpyPadMode) if mode is not None else None
        self.pad_opts = pad_opts

    def __call__(self, array: np.ndarray):
        """
        Args:
            array: the image to generate patches from.
        """
        yield from iter_patch(
            array,
            patch_size=self.patch_size,  # type: ignore
            start_pos=self.start_pos,
            overlap=self.overlap,
            copy_back=False,
            mode=self.mode,
            **self.pad_opts,
        )
