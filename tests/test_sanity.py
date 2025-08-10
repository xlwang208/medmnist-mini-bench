
def test_import_and_versions():
    import bench
    assert hasattr(bench, "get_dataloaders")

def test_dataloader_small():
    from bench import get_dataloaders
    train, val, test, meta = get_dataloaders("pathmnist", batch_size=8, num_workers=0, limit_samples=32)
    x, y = next(iter(train))
    assert x.shape[0] == 8
    assert x.shape[1] in (1, 3)
