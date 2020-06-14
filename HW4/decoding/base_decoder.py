from collections import defaultdict
from typing import Dict, Optional


class BaseDecoder:
    def __init__(self, ref_text_statistics: Dict[str, float], token_len: int):
        self.popularity_token_mapping = {
            i: k for i, (k, _) in enumerate(sorted(ref_text_statistics.items(), key=lambda x: -x[1]))
        }
        self.token_len = token_len
        self._target_stat: Optional[Dict[str, int]] = None

    def decode(self, *args, **kwargs) -> str:
        raise NotImplemented

    @classmethod
    def init_decoder_with_reference_text(cls, text: str, token_len: int):
        ref_text_statistics = cls._count_text_statistics(text, token_len)
        return cls(ref_text_statistics=ref_text_statistics, token_len=token_len)

    @staticmethod
    def _count_text_statistics(text: str, token_len: int) -> Dict[str, int]:
        assert len(text) > 0, "input text is empty"
        assert isinstance(token_len, int), "length of token should be a positive integer"
        assert token_len > 0, "length of token should be positive"

        freq_count = defaultdict(int)
        for i in range(len(text) - token_len + 1):
            ngram = text[i: i + token_len]
            freq_count[ngram] += 1
        return freq_count

    @staticmethod
    def _decode_with_key(target_text: str, key: Dict[str, str]) -> str:
        tokens_to_decode_lens = [len(token) for token in key.keys()]
        assert min(tokens_to_decode_lens) == max(tokens_to_decode_lens), "key must map tokens of same length"

        token_len = tokens_to_decode_lens[0]
        decoded_tokens = []
        for i in range(0, len(target_text) - token_len + 1, token_len):
            token_to_decode = target_text[i: i + token_len]
            decoded_tokens.append(key.get(token_to_decode, token_to_decode))
        decoded_text = ''.join(decoded_tokens)
        return decoded_text
