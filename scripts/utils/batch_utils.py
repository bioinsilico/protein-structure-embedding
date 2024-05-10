import torch


def seq_embeddings_collator(device):
    def __collate_seq_embeddings(batch_list):
        """
        Pads the tensors in a batch to the same size.

        Args:
            batch_list: A list of samples.

        Returns:
            A batch tensor.
        """
        max_len = max([len(sample) for sample in batch_list])
        dim = batch_list[0].shape[1]
        padded_batch = []
        mask_batch = []
        for sample in batch_list:
            padded_sample = torch.zeros([max_len, dim])
            padded_sample[:len(sample)] = sample
            padded_batch.append(padded_sample)

            mask_sample = torch.ones([max_len])
            mask_sample[:len(sample)] = torch.zeros([len(sample)])
            mask_batch.append(mask_sample.bool())

        return torch.stack(padded_batch, dim=0), torch.stack(mask_batch, dim=0).to(device)

    return __collate_seq_embeddings
