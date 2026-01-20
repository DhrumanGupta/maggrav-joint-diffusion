from zarr_dataset import NoddyverseZarrDataset

ds = NoddyverseZarrDataset(
    "/scratch/dhruman_gupta/noddyverse_preprocessed/output-final",
    include_metadata=False,
)

sample = ds[0]

for key, value in sample.items():
    print(key)
    print(value.shape)
    print(value.dtype)
    print("")
