from typing import Tuple, List, Optional

from transformers import MBartTokenizer
import mojimoji

from .utils import SpmMixin, TokenizeHandlerMeta


class MBartTokenizerHandler(MBartTokenizer, SpmMixin, TokenizeHandlerMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, src_lang='ja_XX', **kwargs)
        self._pad_id = self._convert_token_to_id(self.pad_token)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_tokenized_tokens(
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
        is_intermediate_list = []
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
                if i == 0:
                    all_tokens.append(word)
                    tok_to_orig_index.append(None)
                    is_intermediate_list.append(True)

            elif is_before_special or word.startswith(SPM_SPECIAL_TOKEN):
                # Token after single '▁' or '▁word'
                # Head word
                is_before_special = False
                word_index += 1
                orig_to_tok_index.append(len(all_tokens))
                tok_to_orig_index.append(word_index)
                all_tokens.append(word)
                is_intermediate_list.append(False)
            else:
                is_before_special = False
                tok_to_orig_index.append(word_index)
                all_tokens.append(word)
                is_intermediate_list.append(True)

        # </s> token
        all_tokens.append('</s>')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        # 'ja_XX' token
        all_tokens.append('ja_XX')
        tok_to_orig_index.append(None)
        is_intermediate_list.append(True)

        return all_tokens, tok_to_orig_index, orig_to_tok_index, is_intermediate_list
