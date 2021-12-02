from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace,WhitespaceSplit


def get_tokenizer_bpe(data, vocab_size):
    # Configure tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer=normalizers.Sequence([NFD(),StripAccents(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer_src = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]","[EOS]"])

    # Configure batch iterators to train tokenizers from memory
    def batch_iterator_src(batch_size=10000):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]['translation_src']
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]['translation_trg']

    # Train tokenizers
    tokenizer.train_from_iterator(batch_iterator_src(), trainer=trainer_src, length=len(data))

    # Configure postprocessing to add [BOS] and [EOS] tokens to sequences
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", 2),
            ("[EOS]", 3),
        ],
    )
    return tokenizer

def get_tokenizer_wordlevel(data, vocab_size):
    # Configure tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer=normalizers.Sequence([NFD(),StripAccents(), Lowercase()])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer_src = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]","[EOS]"]) 

    # Configure batch iterators to train tokenizers from memory
    def batch_iterator_src(batch_size=10000):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]['translation_src']
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]['translation_trg']

    # Train tokenizers
    tokenizer.train_from_iterator(batch_iterator_src(), trainer=trainer_src, length=len(data))

    # Configure postprocessing to add [BOS] and [EOS] tokens to trg sequence
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", 2),
            ("[EOS]", 3),
        ],
    )

    return tokenizer