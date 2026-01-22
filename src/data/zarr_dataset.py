import json
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class NoddyverseZarrDataset(Dataset):
    """PyTorch dataset for concatenated Noddyverse Zarr stores."""

    def __init__(
        self,
        zarr_path: str,
        fields: Sequence[str] = ("rock_types", "mag", "grv"),
        include_metadata: bool = False,
        decode_metadata: bool = True,
        return_tensors: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.zarr_path = zarr_path
        self.fields = tuple(fields)
        self.include_metadata = include_metadata
        self.decode_metadata = decode_metadata
        self.return_tensors = return_tensors
        self.transform = transform
        self._required_fields = tuple(sorted(set(self.fields) | {"rock_types"}))

        self._group: Optional[zarr.Group] = None
        self._arrays: Dict[str, zarr.Array] = {}
        self._meta_array: Optional[zarr.Array] = None
        self._meta_attr: Optional[str] = None
        self._meta_list: Optional[Sequence[Any]] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_group"] = None
        state["_arrays"] = {}
        state["_meta_array"] = None
        return state

    def _ensure_open(self) -> None:
        if self._group is not None:
            return
        self._group = zarr.open_group(self.zarr_path, mode="r")
        missing = [name for name in self._required_fields if name not in self._group]
        if missing:
            raise ValueError(f"Missing required arrays in Zarr: {missing}")
        self._arrays = {name: self._group[name] for name in self._required_fields}
        if "samples_metadata" in self._group:
            self._meta_array = self._group["samples_metadata"]
        else:
            self._meta_array = None
            self._meta_attr = self._group.attrs.get("samples_metadata")
            self._meta_list = None
        if self._meta_array is None and not self._meta_attr:
            raise ValueError("samples_metadata not found in Zarr store.")

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._arrays["rock_types"].shape[0])

    def _load_metadata(self, idx: int) -> Any:
        if self._meta_array is not None:
            meta_raw = self._meta_array[idx]
        elif self._meta_attr:
            if self._meta_list is None:
                parsed = json.loads(self._meta_attr)
                if not isinstance(parsed, list):
                    raise ValueError("samples_metadata attribute is not a list.")
                self._meta_list = parsed
            meta_raw = self._meta_list[idx]
        else:
            meta_raw = None

        if not self.decode_metadata or meta_raw is None:
            return meta_raw
        if isinstance(meta_raw, str):
            return json.loads(meta_raw)
        return meta_raw

    def _to_tensor(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value

    def _map_properties(
        self, rock_types: np.ndarray, properties: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not properties:
            raise ValueError("Missing properties in metadata.")
        max_id = max(int(key) for key in properties.keys())
        density_lut = np.zeros(max_id + 1, dtype=np.float32)
        susc_lut = np.zeros(max_id + 1, dtype=np.float32)
        for key, values in properties.items():
            rid = int(key)
            try:
                density_lut[rid] = float(values[0])
                susc_lut[rid] = float(values[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid properties entry for rock id {rid}: {values!r}"
                ) from exc
        rock_clipped = np.clip(rock_types, 0, max_id).astype(np.int32, copy=False)
        density = density_lut[rock_clipped]
        susceptibility = susc_lut[rock_clipped]
        return density, susceptibility

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()
        rock_types = self._arrays["rock_types"][idx]
        mag = self._arrays["mag"][idx] if "mag" in self._arrays else None
        grv = self._arrays["grv"][idx] if "grv" in self._arrays else None

        meta = self._load_metadata(idx)
        if isinstance(meta, np.generic):
            meta = meta.item()
        if isinstance(meta, np.ndarray):
            if meta.shape == ():
                meta = meta.item()
            else:
                raise ValueError(
                    f"Unexpected metadata array at index {idx}: shape={meta.shape}"
                )
        if isinstance(meta, str):
            meta = json.loads(meta)
        if not isinstance(meta, dict):
            raise ValueError(f"Unexpected metadata format at index {idx}: {meta!r}")
        properties = meta.get("properties", {})
        density, susceptibility = self._map_properties(rock_types, properties)

        sample: Dict[str, Any] = {
            "density": density,
            "susceptibility": susceptibility,
        }
        if mag is not None:
            sample["mag"] = mag
        if grv is not None:
            sample["grv"] = grv
        if self.include_metadata:
            sample["metadata"] = meta
        if self.return_tensors:
            sample = {key: self._to_tensor(val) for key, val in sample.items()}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
