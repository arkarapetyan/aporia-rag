"""The authors a question names, so that retrieval can keep to them.

Plain code, not a model: the corpus has a short, fixed list of authors, and a
surname is either in the question or not. An author counts as named when the
question contains the surname (the last word of the name) as a whole word, in
any case: "kant", "Kant's" and "Immanuel Kant" all name Immanuel Kant, while
"Kantian" names nobody. What a whole-word match cannot tell apart, such as a
lowercase "mill" that is not John Stuart Mill, is rare in questions about
philosophy; the CLI prints whom it searched, so a wrong match is visible.
"""

import re
from collections.abc import Iterable


def authors_named_in(question: str, authors: Iterable[str]) -> list[str]:
    """The `authors` the question names by surname, in the order given."""
    return [author for author in authors if _surname(author).search(question)]


def _surname(author: str) -> re.Pattern[str]:
    surname = author.rsplit(maxsplit=1)[-1]
    return re.compile(rf"\b{re.escape(surname)}\b", re.IGNORECASE)
