class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 orig_to_tok_index,
                 tok_to_orig_index,
                 input_ids,
                 input_mask,
                 segment_ids,
                 heads=None,
                 arguments_set=None,
                 ng_arg_ids_set=None,
                 token_tag_indices=None,
                 spans=None,
                 span_labels=None,
                 is_mention_labels=None,
                 metadata=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.orig_to_tok_index = orig_to_tok_index
        self.tok_to_orig_index = tok_to_orig_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.heads = heads
        self.arguments_set = arguments_set
        self.ng_arg_ids_set = ng_arg_ids_set
        self.token_tag_indices = token_tag_indices
        self.spans = spans
        self.span_labels = span_labels
        self.is_mention_labels = is_mention_labels
        self.metadata = metadata
