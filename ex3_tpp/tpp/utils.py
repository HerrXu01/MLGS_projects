from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def get_sequence_batch(
    inter_event_times: List[TensorType[torch.float32]],
) -> Tuple[
    TensorType[torch.float32, "batch", "max_seq_length"],
    TensorType[torch.bool, "batch", "max_seq_length"],
]:
    """
    Generate padded batch and mask for list of sequences.

        Args:
            inter_event_times (List): list of inter-event times

        Returns:
            batch: batched inter-event times. shape [batch_size, max_seq_length]
            mask: boolean mask indicating inter-event times. shape [batch_size, max_seq_length]
    """

    #######################################################
    # write here
    batch = pad_sequence(inter_event_times, batch_first=True)
    # in case there are 0s in the original inter_event_times
    processed_list = [torch.where(tensor != 0.0, tensor, torch.tensor(1.0)) for tensor in inter_event_times]
    processed_tensor = pad_sequence(processed_list, batch_first=True)
    mask = (processed_tensor != 0.0)
    #######################################################

    return batch, mask


@typechecked
def get_tau(
    t: TensorType[torch.float32, "sequence_length"], t_end: TensorType[torch.float32, 1]
) -> TensorType[torch.float32]:
    """
    Compute inter-eventtimes from arrival times

        Args:
            t: arrival times. shape [seq_length]
            t_end: end time of the temporal point process.

        Returns:
            tau: inter-eventtimes.
    """
    # compute inter-eventtimes
    #######################################################
    # write here
    t2 = torch.cat((t, t_end), dim=0)
    t_0 = torch.tensor([0], dtype=torch.float32)
    t1 = torch.cat((t_0, t), dim=0)
    tau = t2 - t1
    #######################################################

    return tau
