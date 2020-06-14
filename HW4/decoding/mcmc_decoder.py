from math import log
from random import uniform, sample
from typing import Dict, Optional, Union

import numpy as np

from decoding.base_decoder import BaseDecoder


class MCMCDecoder(BaseDecoder):
    def __init__(self, ref_text_statistics: Dict[str, int], token_len: int):
        super().__init__(ref_text_statistics, token_len)
        # make a high-level copy
        self._ref_stat = {
            token: freq
            for token, freq in ref_text_statistics.items()
        }

    def decode(
            self,
            target_text: str,
            n_draws: int = 10_000,
            early_stopping_rounds: Optional[int] = None,
            scaling_factor: float = 1.0,
            verbose: Union[bool, int] = 2_000,
    ) -> str:
        key = self._run_mcmc_loop(target_text, n_draws, early_stopping_rounds, scaling_factor, verbose)
        decoded_text = self._decode_with_key(target_text, key)
        return decoded_text

    def _run_mcmc_loop(
            self,
            target_text: str,
            n_draws: int,
            early_stopping_rounds: Optional[int],
            scaling_factor: float,
            verbose: Union[bool, int],
    ) -> Dict[str, str]:
        target_text_stat = super()._count_text_statistics(target_text, self.token_len)
        old_key = self._generate_random_key(target_text_stat)
        old_decoded_text = self._decode_with_key(target_text, old_key)
        old_decoded_text_stat = super()._count_text_statistics(old_decoded_text, self.token_len)
        old_score = self._estimate_score(old_decoded_text_stat)

        best_score = old_score
        best_key = old_key
        best_decoded_text = old_decoded_text
        best_step_num = 0

        for step_num in range(n_draws):
            new_key = self._sample_new_key(old_key)
            new_decoded_text = self._decode_with_key(target_text, new_key)
            new_decoded_text_stat = super()._count_text_statistics(new_decoded_text, self.token_len)
            new_score = self._estimate_score(new_decoded_text_stat)

            if self._accept_proposed_key(old_score, new_score, scaling_factor):
                old_key = new_key
                old_score = new_score

                if old_score > best_score:
                    best_score = new_score
                    best_key = new_key
                    best_decoded_text = new_decoded_text
                    best_step_num = step_num

            if verbose:
                if (step_num % verbose) == 0:
                    print(f"step_number: {step_num}")
                    print(f"likelihood: {best_score}")
                    print(best_decoded_text, "\n\n", sep="")

            if early_stopping_rounds:
                if (step_num - best_step_num) > early_stopping_rounds:
                    print(f"MCMC is stopped: score hasn't been improved for {early_stopping_rounds} steps")
                    break

        return best_key

    def _generate_random_key(self, target_text_stat: Dict):
        target_letters = list(set("".join(target_text_stat.keys())))
        ref_letters = list(set("".join(self._ref_stat.keys())))

        if len(target_letters) > len(ref_letters):
            target_letters = target_letters[:len(ref_letters)]

        random_key = {
            target_letter: ref_letter for target_letter, ref_letter in
            zip(target_letters, sample(ref_letters, k=len(target_letters)))
        }
        return random_key

    @staticmethod
    def _sample_new_key(old_key: Dict[str, str]) -> Dict[str, str]:
        new_key = {k: v for k, v in old_key.items()}

        token_1, token_2 = sample(list(new_key), 2)
        new_key[token_1], new_key[token_2] = new_key[token_2], new_key[token_1]
        return new_key

    def _estimate_score(self, text_stat: Dict[str, float]) -> float:
        tokens = text_stat.keys()
        tokens_num = len(tokens)

        r = np.ones(tokens_num, dtype=np.float32)
        f = np.ones(tokens_num, dtype=np.float32)

        for i, token in enumerate(tokens):
            f[i] += text_stat.get(token, 0.0)
            r[i] += self._ref_stat.get(token, 0.0)
        score = np.sum(f * np.log(r))
        return float(score)

    @staticmethod
    def _accept_proposed_key(old_score: float, new_score: float, scaling_factor: float) -> bool:
        if new_score > old_score:
            return True

        u = uniform(0.0, 1.0)
        thr = (new_score - old_score) * scaling_factor
        return log(u) < thr
