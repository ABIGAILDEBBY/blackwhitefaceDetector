import splitfolders


def data_split(
    input_folder: str,
    output_folder: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    # Ratio of split are in order of train/val/test.
    splitfolders.ratio(
        f"{input_folder}",
        output=f"{output_folder}",
        seed=42,
        ratio=(train_ratio, test_ratio, val_ratio),
    )
