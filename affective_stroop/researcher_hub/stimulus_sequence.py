# researcher_hub/stimulus_sequence.py

from sweetbean import Experiment, Block
from sweetbean.stimulus import Text
from sweetbean.variable import TimelineVariable
from design import create_emotion_stroop_trials

def stimulus_sequence(n_train):
    timeline = create_emotion_stroop_trials(num_trials=n_train)

    color_var = TimelineVariable("color")
    word_var = TimelineVariable("word")
    valence_var = TimelineVariable("valence")

    instructions = Text(text="Name the ink color of each word. Press SPACE to start.", choices=[" "])
    stroop = Text(duration=2000, text=word_var, color=color_var, choices=["c", "d", "n", "j"])
    instr_block = Block([instructions])
    task_block = Block([stroop], timeline)
    exit_block = Block([Text(duration=2000, text="Thanks!", choices=[])])

    experiment = Experiment([instr_block, task_block, exit_block])
    return experiment.to_js_string(as_function=True, is_async=True)
