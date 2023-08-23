import os
import sys

sys.path.append(os.path.dirname(__file__))

from tagger import model


def tag(sentence, reload_model=True):
    return model.predict(sentence, reload_model)
