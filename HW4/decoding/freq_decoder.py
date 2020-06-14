from typing import Dict

from decoding.base_decoder import BaseDecoder


class FreqDecoder(BaseDecoder):
    def __init__(self, ref_text_statistics: Dict[str, int], token_len: int):
        super().__init__(ref_text_statistics, token_len)

    def decode(self, target_text: str) -> str:
        target_text_stat = super()._count_text_statistics(target_text, self.token_len)
        key = self._generate_key(target_text_stat)
        decoded_text = self._decode_with_key(target_text, key)
        return decoded_text

    def _generate_key(self, target_text_stat: Dict[str, float]) -> Dict[str, str]:
        token_popularity_mapping = {
            k: i for i, (k, _) in enumerate(sorted(target_text_stat.items(), key=lambda x: -x[1]))
        }

        key = {}
        for cipher_token, cipher_token_popularity in token_popularity_mapping.items():
            try:
                key[cipher_token] = self.popularity_token_mapping[cipher_token_popularity]
            except KeyError:
                continue
        return key
