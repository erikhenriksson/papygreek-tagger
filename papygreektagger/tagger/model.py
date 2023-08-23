import os
import sys
from math import floor
import unicodedata
import regex as re
from flair.models import SequenceTagger
from flair.data import Sentence

from .rules import word_classes

os.chdir(os.path.dirname(__file__))
tagger = SequenceTagger.load("v4/best-model.pt")
os.chdir(sys.path[0])

pad = lambda x, y, filler: x + [filler] * (len(y) - len(x))
plain = lambda s: "".join([unicodedata.normalize("NFD", a)[0].lower() for a in s])
numeral = lambda x: "num" if x else ""
just_greek = lambda x: re.sub(r"\p{^Greek}", "", (x or ""))
punctuation = lambda x: x if x in ",..·;;:·." else "αβγδεφηιξκλμ"
two_decimals = lambda x: round(floor(x * 100) / 100, 2)


def preformat(sentence, version):
    return [
        numeral(x[f"{version}_num"])
        or (x[f"{version}_plain"] or "").replace("_", "")
        or punctuation(f"{version}_plain")
        for x in sentence
    ]


def afterformat(prediction, confidence, token, token_plain):
    # Semi-manual confidence is 0.99 (default is 1
    conf = 0.99

    # Unknown postag, revert to defaults (None, confidence 1)
    if prediction in ["_", "<unk>", "0", ""]:
        return None, 1

    # Punct
    if token_plain in ",..·;;:·.":
        return "u--------", conf

    # Tokens with diacritics
    if token == "ὧν":
        return "p-p---mg-", conf
    elif token == "ἀλλά":
        return "c--------", conf
    elif token == "ὅτι":
        return "c--------", conf

    # Word classes
    elif token_plain in word_classes["CONJUNCTIVES"]:
        return "c--------", conf
    elif token_plain in word_classes["ADVERBS"]:
        return "d--------", conf
    elif token_plain in word_classes["PREPOSITIONS"]:
        return "r--------", conf
    elif token_plain in word_classes["NOUNS"]:
        return "n-s------", conf

    # Manual tweaks
    if token_plain == "εγω":
        return "p1s---mn-", conf
    if token_plain == "μου":
        return "p1s----g-", conf
    if token_plain == "με":
        return "p1s----a-", conf
    if token_plain == "εμε":
        return "p1s----a-", conf
    if token_plain == "μοι":
        return "p1s----d-", conf
    if token_plain == "υμων":
        return "p2p---mg-", conf
    if token_plain == "σε":
        return "p2s----a-", conf
    if token_plain == "σοι":
        return "p2s----d-", conf
    if token_plain == "υμας":
        return "p2p----a-", conf
    if token_plain == "υμιν":
        return "p2p----d-", conf
    if token_plain == "σου":
        return "p2s----g-", conf
    if token_plain == "συ":
        return "p2s---mn-", conf
    if token_plain == "ημιν":
        return "p1p---md-", conf
    if token_plain == "ημεας":
        return "p1p---ma-", conf
    if token_plain == "ημας":
        return "p1p---ma-", conf
    if token_plain == "ημεις":
        return "p1p---mn-", conf
    if token_plain == "ημων":
        return "p1p---mg-", conf
    if token_plain == "σεαυτου":
        return "p2s---mg-", conf
    if token_plain == "εμοι":
        return "p1s---md-", conf
    if token_plain == "ημειν":
        return "p-p---md-", conf
    if token_plain == "ταελολους":
        return "n-s---fn-", conf
    if token_plain == "ασπασαι":
        return "v2same---", conf
    if token_plain == "πυρου":
        return "n-s---mg-", conf
    if token_plain == "βουβαστω":
        return "n-s---md-", conf
    if token_plain == "επιτροπου":
        return "a-s---mg-", conf
    if token_plain == "δει":
        return "v3spia---", conf

    # As a last resort, fix parts of postag
    prediction = list(prediction)

    if prediction[0] == "b":
        prediction[0] = "c"
    if prediction[0] == "i":
        prediction[0] = "m"
    if len(prediction) > 6 and prediction[6] == "c":
        prediction[6] = "m"

    # Numeral
    if token_plain == "num":
        prediction[0] = "m"

    return "".join(prediction), confidence


def predict(sentence):
    for version in ["reg", "orig"]:
        version_tokens = preformat(sentence, version)
        version_sentence = Sentence(
            " ".join(version_tokens),
            use_tokenizer=False,
        )
        tagger.predict(version_sentence)

        version_result = pad(
            version_sentence.to_dict()["all labels"],
            version_tokens,
            {"value": "<unk>", "confidence": "0.99"},
        )

        for i, _ in enumerate(sentence):
            token = sentence[i]
            version_pred = version_result[i]

            version_value, version_confidence = afterformat(
                version_pred["value"],
                version_pred["confidence"],
                just_greek(token[f"{version}_form"]),
                version_tokens[i],
            )

            token[f"{version}_postag"] = version_value
            token[f"{version}_postag_confidence"] = two_decimals(version_confidence)
            token[f"{version}_postag"] = version_value
            token[f"{version}_postag_confidence"] = two_decimals(version_confidence)

            for flair_token in version_sentence:
                flair_token.clear_embeddings()

    return sentence
