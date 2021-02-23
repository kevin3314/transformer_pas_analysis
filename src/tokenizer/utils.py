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
    def tokenize_content_words(
        self,
        words: List[str],
        all_tokens: List[str],
        tok_to_orig_index: List[int],
        orig_to_tok_index: List[int],
        is_intermediate_list: List[bool],
        remove_alone_special: bool = False
    ) -> None:
        """Process on words and append to several lists.

        If remove_alone_special, single "▁" is removed except head one.
        This option is for mBART, which does not employ morpheme split.

        Args:
            words (List[str]): Word to process.
            all_tokens (List[str]): Tokens, including special ones.
            tok_to_orig_index (List[int]): List from token idx to original word idx.
            orig_to_tok_index (List[int]): List from original words idx to token.
            is_intermediate_list (List[bool]): List of whether token in idx is intermediate or not
            remove_alone_special (bool): Remove alone special token or not.
        """
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

        word_index = -1
        is_before_special = False
        tokens = [token for word in words for token in self.tokenize(word)]

        if len(words) != _count_special_token(tokens):
            tokens = _fix_wierd_tokenize(words, tokens)

        for i, word in enumerate(tokens):
            if word == SPM_SPECIAL_TOKEN:
                # Single '▁'
                assert not is_before_special
                is_before_special = True

                # If remove_alone_special and index is not 0, skip it
                if remove_alone_special or i != 0:
                    continue
                tok_to_orig_index.append(None)
                is_intermediate_list.append(True)

            elif is_before_special or word.startswith(SPM_SPECIAL_TOKEN):
                # Token after single '▁' or '▁word'
                # Head word
                is_before_special = False
                word_index += 1
                orig_to_tok_index.append(len(all_tokens))
                tok_to_orig_index.append(word_index)
                is_intermediate_list.append(False)
            else:
                is_before_special = False
                tok_to_orig_index.append(word_index)
                is_intermediate_list.append(True)

            # Unless (remove_alone_special and index is not 0), append token
            all_tokens.append(word)

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
