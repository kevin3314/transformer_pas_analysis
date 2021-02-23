from typing import Tuple, List, Optional

from transformers import BartSPMTokenizer
import mojimoji

from .utils import SpmMixin, EncDecTokenizeHandlerMeta


class BartSPMTokenizeHandler(BartSPMTokenizer, SpmMixin, EncDecTokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_encoder_tokenized_tokens(
            self, words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        """TODO: Insert </s> between each sentences.
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

        all_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        is_intermediate_list = []
        word_index = -1
        is_before_special = False

        sentence = " ".join(words)
        tokenized_sentence = self.tokenize(sentence)

        # <s> token
        all_tokens.append('<s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        if len(words) != _count_special_token(tokenized_sentence):
            tokenized_sentence = _fix_wierd_tokenize(words, tokenized_sentence)

        for word in tokenized_sentence:
            if word == SPM_SPECIAL_TOKEN:
                # Single '▁'
                assert not is_before_special
                is_before_special = True
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
            all_tokens.append(word)

            # if word == '▁。':
            #     all_tokens.append('</s>')
            #     tok_to_orig_index.append(None)
            #     is_intermediate_list.append(True)

        # </s> token
        all_tokens.append('</s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        assert (len(words) -
                1 == word_index), f"{len(words) - 1} != {word_index}\n{words}"
        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list

    def get_decoder_tokenized_tokens(
        self,
        words: List[str]
    ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        return self.get_encoder_tokenized_tokens(words)
