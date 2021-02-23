from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import mojimoji


class TokenizeHandlerMeta(ABC):
    @abstractmethod
    def get_encoder_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        pass

    @abstractmethod
    def check_intermediate(self, word: str) -> bool:
        pass

    @abstractmethod
    def check_intermediate_tokens(self, all_tokens: List[str]) -> List[bool]:
        pass

    @property
    @abstractmethod
    def pad_id(self) -> int:
        pass


class EncDecTokenizeHandlerMeta(TokenizeHandlerMeta):
    @abstractmethod
    def get_decoder_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        pass


class SpmMixin:
    def get_encoder_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        SPM_SPECIAL_TOKEN = "▁"

        def _count_special_token(words):
            """Count SPM_SPECIAL_TOKEN in words.
            """
            count = 0
            for word in words:
                if SPM_SPECIAL_TOKEN in word:
                    count += 1
            return count

        def _fix_wierd_tokenize(words, tokenized_sentence):
            """Fix weird behavior of tokenizer.
            Ex. 'ab´db' -> '▁ab' + '▁´' + 'db'
            """
            result = []
            idx = 0
            for word in words:
                found_special = False
                # Scan tokens in tokenized_sentence.
                # If find weird SPM_SPECIAL_TOKEN, remove them.
                read_token = ""
                while 1:
                    current_token = tokenized_sentence[idx]
                    tmp_token = mojimoji.han_to_zen(
                        current_token.replace(SPM_SPECIAL_TOKEN, ""))
                    read_token += tmp_token
                    if found_special:
                        # Intermediate, remove SPM_SPECIAL_TOKEN
                        result.append(tmp_token)
                    else:
                        # Head, keep original token
                        result.append(tokenized_sentence[idx])
                    if current_token.startswith(SPM_SPECIAL_TOKEN):
                        found_special = True
                    idx += 1
                    if len(read_token) == len(word):
                        break
            return result

        all_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        word_index = -1
        is_before_special = False

        sentence = " ".join(words)
        tokenized_sentence = self.tokenize(sentence)

        if len(words) != _count_special_token(tokenized_sentence):
            tokenized_sentence = _fix_wierd_tokenize(words, tokenized_sentence)

        for word in tokenized_sentence:
            if word == SPM_SPECIAL_TOKEN:
                # Single '▁'
                assert not is_before_special
                is_before_special = True
                tok_to_orig_index.append(None)
            elif is_before_special or word.startswith(SPM_SPECIAL_TOKEN):
                # Token after single '▁' or '▁word'
                # Head word
                is_before_special = False
                word_index += 1
                orig_to_tok_index.append(len(all_tokens))
                tok_to_orig_index.append(word_index)
            else:
                is_before_special = False
                tok_to_orig_index.append(word_index)
            all_tokens.append(word)

        assert (len(words) -
                1 == word_index), f"{len(words) - 1} != {word_index}\n{words}"

        is_intermediate_list = self.check_intermediate_tokens(all_tokens)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list

    # TODO: fix this function somehow
    # Example: '今日　は　雨　です' -> ['▁', '今日', '▁', 'は', '▁', '雨', '▁', 'です']
    def check_intermediate(self, word: str) -> bool:
        return word.startswith("▁")

    def check_intermediate_tokens(self, all_tokens: List[str]) -> List[bool]:
        """各トークンが単語の中間か、それ以外かを表す bool 値のリストを返す

        Args:
            all_tokens (List[str]): トークンのリスト

        Returns:
            List[bool]: 各トークンが単語の中間か、それ以外かを表すリスト
                        中間なら True, それ以外なら False
        """
        head = all_tokens[0].startswith("▁")
        tail = [
            prev == "▁" or current.startswith("▁")
            for prev, current in zip(all_tokens, all_tokens[1:])
        ]
        assert len([head] + tail) == len(all_tokens)
        return [not (bo) for bo in ([head] + tail)]
