import pickle
from monai import config
from monai.data.meta_obj import MetaObj
from collections.abc import Mapping, Sequence
from torch.utils.data._utils.collate import default_collate
from monai.utils import (
    first,
    TraceKeys
)

PICKLE_KEY_SUFFIX = TraceKeys.KEY_SUFFIX


def pickle_operations(data, key=PICKLE_KEY_SUFFIX, is_encode: bool = True):
    """
    Applied_operations are dictionaries with varying sizes, this method converts them to bytes so that we can (de-)collate.

    Args:
        data: a list or dictionary with substructures to be pickled/unpickled.
        key: the key suffix for the target substructures, defaults to "_transforms" (`data.utils.PICKLE_KEY_SUFFIX`).
        is_encode: whether it's encoding using pickle.dumps (True) or decoding using pickle.loads (False).
    """
    if isinstance(data, Mapping):
        data = dict(data)
        for k in data:
            if f"{k}".endswith(key):
                if is_encode and not isinstance(data[k], bytes):
                    data[k] = pickle.dumps(data[k], 0)
                if not is_encode and isinstance(data[k], bytes):
                    data[k] = pickle.loads(data[k])
        return {k: pickle_operations(v, key=key, is_encode=is_encode) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [pickle_operations(item, key=key, is_encode=is_encode) for item in data]
    return data


def collate_meta_tensor(batch):
    """collate a sequence of meta tensor sequences/dictionaries into
    a single batched metatensor or a dictionary of batched metatensor"""
    if not isinstance(batch, Sequence):
        raise NotImplementedError()
    elem_0 = first(batch)
    if isinstance(elem_0, MetaObj):
        collated = default_collate(batch)
        collated.applied_operations = [i.applied_operations or TraceKeys.NONE for i in batch]
        # collated.meta = default_collate([i.meta or TraceKeys.NONE for i in batch])
        collated.is_batch = True
        return collated
    if isinstance(elem_0, Mapping):
        return {k: collate_meta_tensor([d[k] for d in batch if k in d]) for k in elem_0}
    if isinstance(elem_0, (tuple, list)):
        return [collate_meta_tensor([d[i] for d in batch]) for i in range(len(elem_0))]

    # no more recursive search for MetaTensor
    return default_collate(batch)


def list_data_collate(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    key = None
    try:
        if config.USE_META_DICT:
            data = pickle_operations(data)  # bc 0.9.0
        if isinstance(elem, Mapping):
            ret = {}
            for k in elem:
                key = k
                data_for_batch = [d[key] for d in data]
                ret[key] = collate_meta_tensor(data_for_batch)
        else:
            ret = collate_meta_tensor(data)
        return ret
    except RuntimeError as re:
        re_str = str(re)
        if "equal size" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                    "\n\nMONAI hint: if your transforms intentionally create images of different shapes, creating your "
                    + "`DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem (check its "
                    + "documentation)."
            )
        raise RuntimeError(re_str) from re
    except TypeError as re:
        re_str = str(re)
        if "numpy" in re_str and "Tensor" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                    "\n\nMONAI hint: if your transforms intentionally create mixtures of torch Tensor and numpy ndarray, "
                    + "creating your `DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem "
                    + "(check its documentation)."
            )
        raise TypeError(re_str) from re
