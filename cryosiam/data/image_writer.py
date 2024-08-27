import mrcfile
import tifffile
import numpy as np
from monai.data import ImageWriter
from typing import Mapping, Optional
from monai.config import DtypeLike, NdarrayOrTensor, PathLike


class TiffWriter(ImageWriter):
    """
    Write data into files on disk using TIFFFILE.
    .. code-block:: python
        import numpy as np
        from cryosiam.data import TiffWriter
        np_data = np.arange(48).reshape(3, 4, 4)
        writer = TiffWriter()
        writer.set_data_array(np_data, channel_dim=None)
        writer.write("test1.tiff", verbose=True)
    """

    def __init__(self, output_dtype: DtypeLike = np.float32, **kwargs):
        super().__init__(output_dtype=output_dtype, **kwargs)

    def set_data_array(self, data_array: NdarrayOrTensor, channel_dim: Optional[int] = 0,
                       squeeze_end_dims: bool = True, **kwargs):
        """
        Convert ``data_array`` into 'channel-last' numpy ndarray.
        Args:
            data_array: input data array with the channel dimension specified by ``channel_dim``.
            channel_dim: channel dimension of the data array. Defaults to 0.
                ``None`` indicates data without any channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed.
            kwargs: keyword arguments passed to ``self.convert_to_channel_last``,
                currently support ``spatial_ndim``, defauting to ``3``.
        """
        self.data_obj = self.convert_to_channel_last(
            data=data_array,
            channel_dim=channel_dim,
            squeeze_end_dims=squeeze_end_dims,
            spatial_ndim=kwargs.pop("spatial_ndim", 3),
        )

    def write(self, filename: PathLike, verbose: bool = False, **obj_kwargs):
        """
        Save the data with TIFFFILE into the given filename.
        Args:
            filename: filename or PathLike object.
            verbose: if ``True``, log the progress.
        """
        super().write(filename, verbose=verbose)
        tifffile.imwrite(filename, self.data_obj)


class MrcWriter(ImageWriter):
    """
    Write data into files on disk using MRCFILE.
    .. code-block:: python
        import numpy as np
        from cryosiam.data import MrcWriter
        np_data = np.arange(48).reshape(3, 4, 4)
        writer = MrcWriter()
        writer.set_data_array(np_data, channel_dim=None)
        writer.set_metadata({"voxel_size": (1, 1, 1)})
        writer.write("test1.mrc", verbose=True)
    """

    def __init__(self, output_dtype: DtypeLike = np.float32, overwrite=True, **kwargs):
        super().__init__(output_dtype=output_dtype, **kwargs)
        self.overwrite = overwrite

    def set_data_array(self, data_array: NdarrayOrTensor, channel_dim: Optional[int] = 0,
                       squeeze_end_dims: bool = True, **kwargs):
        """
        Convert ``data_array`` into 'channel-last' numpy ndarray.
        Args:
            data_array: input data array with the channel dimension specified by ``channel_dim``.
            channel_dim: channel dimension of the data array. Defaults to 0.
                ``None`` indicates data without any channel dimension.
            squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed.
            kwargs: keyword arguments passed to ``self.convert_to_channel_last``,
                currently support ``spatial_ndim``, defauting to ``3``.
        """
        self.data_obj = self.convert_to_channel_last(
            data=data_array,
            channel_dim=channel_dim,
            squeeze_end_dims=squeeze_end_dims,
            spatial_ndim=kwargs.pop("spatial_ndim", 3),
        )

    def set_metadata(self, meta_dict: Optional[Mapping], **options):
        """
        Set metadata
        Args:
            meta_dict: a metadata dictionary for voxel_size information.
        """
        self.meta_data = meta_dict

    def write(self, filename: PathLike, verbose: bool = False, **obj_kwargs):
        """
        Save the data with MRCFILE into the given filename.
        Args:
            filename: filename or PathLike object.
            verbose: if ``True``, log the progress.
        See also:
            - https://mrcfile.readthedocs.io/en/latest/usage_guide.html#opening-mrc-files
        """
        super().write(filename, verbose=verbose)
        with mrcfile.new(filename, overwrite=self.overwrite) as mrc:
            mrc.set_data(self.data_obj)
            mrc.voxel_size = self.meta_data['voxel_size']
