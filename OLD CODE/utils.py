def iter_indeices(batch, batch_size, dataset_size):
    batch_start = batch * batch_size
    batch_end = (batch + 1) * batch_size
    if batch_end > dataset_size:
        batch_end = dataset_size-1

    return batch_start, batch_end