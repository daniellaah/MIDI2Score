import torch

from midi2score.data import (
    FakeDataConfig,
    FakeSeq2SeqDataset,
    build_fake_dataloader,
    collate_seq2seq_batch,
)


def test_fake_dataset_is_deterministic_per_index() -> None:
    config = FakeDataConfig(
        num_samples=4,
        src_vocab_size=64,
        tgt_vocab_size=80,
        min_source_length=4,
        max_source_length=6,
        min_target_length=3,
        max_target_length=5,
        seed=11,
    )
    dataset = FakeSeq2SeqDataset(config)

    sample_a = dataset[2]
    sample_b = dataset[2]

    assert torch.equal(sample_a["src_tokens"], sample_b["src_tokens"])
    assert torch.equal(sample_a["tgt_tokens"], sample_b["tgt_tokens"])
    assert sample_a["src_tokens"][-1].item() == config.eos_token_id
    assert sample_a["tgt_tokens"][0].item() == config.bos_token_id
    assert sample_a["tgt_tokens"][-1].item() == config.eos_token_id


def test_collate_seq2seq_batch_builds_shifted_targets() -> None:
    config = FakeDataConfig(
        num_samples=3,
        src_vocab_size=64,
        tgt_vocab_size=80,
        min_source_length=4,
        max_source_length=7,
        min_target_length=4,
        max_target_length=7,
        seed=5,
    )
    dataset = FakeSeq2SeqDataset(config)
    examples = [dataset[0], dataset[1]]

    batch = collate_seq2seq_batch(examples, pad_token_id=config.pad_token_id)

    assert batch.src_tokens.shape[0] == 2
    assert batch.tgt_input_tokens.shape == batch.tgt_output_tokens.shape
    assert torch.equal(batch.tgt_input_tokens[:, 1:], batch.tgt_output_tokens[:, :-1])
    assert batch.src_padding_mask.dtype == torch.bool
    assert batch.tgt_padding_mask.dtype == torch.bool


def test_build_fake_dataloader_returns_seq2seq_batch() -> None:
    config = FakeDataConfig(
        num_samples=8,
        src_vocab_size=64,
        tgt_vocab_size=64,
        min_source_length=5,
        max_source_length=8,
        min_target_length=5,
        max_target_length=8,
        seed=3,
    )

    loader = build_fake_dataloader(config, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch.src_tokens.shape[0] == 4
    assert batch.tgt_input_tokens.shape[0] == 4
    assert batch.tgt_output_tokens.shape[0] == 4
