# researcher_hub/design.py

from sweetpea import Factor, DerivedLevel, WithinTrial, CrossBlock, synthesize_trials, experiments_to_dicts

def create_emotion_stroop_trials(num_trials=40):
    colors = ["red", "green", "blue", "yellow"]

    # Word list: emotional vs neutral
    emotion_words = ["death", "fear", "war", "pain"]     # Emotional
    neutral_words = ["table", "book", "window", "chair"] # Neutral

    all_words = emotion_words + neutral_words

    color = Factor("color", colors)
    word = Factor("word", all_words)

    def is_emotional(w): return w in emotion_words

    emotional = DerivedLevel("emotional", WithinTrial(is_emotional, [word]))
    neutral = DerivedLevel("neutral", WithinTrial(lambda w: not is_emotional(w), [word]))
    valence = Factor("valence", [emotional, neutral])

    design = [color, word, valence]
    crossing = [color, word]  # we cross color with word (not with valence)
    block = CrossBlock(design, crossing, [])

    trials = synthesize_trials(block, num_trials)[0]
    return experiments_to_dicts(block, [trials])[0]
