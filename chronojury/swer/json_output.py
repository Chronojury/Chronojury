"""Read model JSON without changing the text inside its narrative fields."""
import json
import re


_TEXT_FIELD = re.compile(
    r'^(\s*"(?:fact|observed_fact|summary|insufficiency_reason|next_retrieval_goal|'
    r'buyer_core_claim|seller_core_claim|dispute_focus)"\s*:\s*")(.*)("\s*,?\s*)$'
)


def load_model_json(text):
    """Accept unescaped quotes only in a standalone narrative-field line.

    The original response stays in the trace. Only JSON escaping is added for
    decoding; every character of the field value is retained. Ambiguous lines
    and other malformed JSON still raise the original parsing error.
    """
    lines = text.splitlines(keepends=True)
    for _ in range(len(lines) + 1):
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            if error.msg != "Expecting ',' delimiter" or error.lineno > len(lines):
                raise
            line = lines[error.lineno - 1]
            match = _TEXT_FIELD.fullmatch(line)
            if match is None:
                raise
            prefix, value, suffix = match.groups()
            if re.search(r'"\s*:\s*|",\s*"', value):
                raise
            escaped = []
            backslashes = 0
            for char in value:
                if char == '"' and backslashes % 2 == 0:
                    escaped.append('\\')
                escaped.append(char)
                backslashes = backslashes + 1 if char == '\\' else 0
            replacement = prefix + ''.join(escaped) + suffix
            if replacement == line:
                raise
            lines[error.lineno - 1] = replacement
            text = ''.join(lines)
    raise ValueError('Model JSON could not be decoded')
