#!/usr/bin/env python3
"""
practice_task.py
Standardised positioning version for cross-screen compatibility.
"""

from psychopy import visual, core, event
import pandas as pd
import os

# Import core loaders and logic from demo_task.py
from demo_task import (
    create_window_and_components, 
    setup_trial_visuals, 
    load_label_map, 
    build_participant_label_pool,
    draw_extras,
    draw_buttons,
    draw_buttons_feedback,
    check_response,
    show_instruction_screen
)

# =========================== LAYOUT CONSTANTS (Standardised) =========================== #
# These units are relative to screen height (0.5 = top, -0.5 = bottom)
IMG_HEIGHT = 0.5
IMG_TOP_EDGE = 0.25  # Since image is centered at (0,0) and is 0.5 high

# Positioning feedback elements in a "Safe Zone" above the image
Y_FEEDBACK_HINT = IMG_TOP_EDGE + 0.03  # 0.28
Y_FEEDBACK_MAIN = Y_FEEDBACK_HINT + 0.07  # 0.35
Y_PROMPT        = Y_FEEDBACK_MAIN + 0.07  # 0.42

# Instructional layout
Y_INSTR_TEXT_HIGH = 0.22
Y_INSTR_TEXT_MID  = 0.10
Y_INSTR_IMAGE     = -0.10
Y_SPACE_PROMPT    = -0.40
Y_BUTTON_LABELS   = -0.38

# =========================== CONFIGURATION =========================== #
PRACTICE_IMG_DIR = "practice_images"
PRACTICE_LABEL_CSV = "practice_labels.csv"

PRACTICE_DESIGN = [
    {'ObjectSpace': '1', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': False},
    {'ObjectSpace': '2', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': False},
    {'ObjectSpace': '1', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': True},
    {'ObjectSpace': '2', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': True},
    
    {'ObjectSpace': '3', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': False},
    {'ObjectSpace': '4', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': False},
    {'ObjectSpace': '3', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': True}, 
    {'ObjectSpace': '4', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': True}
]

def run_custom_practice_trial(win, clock, trial, components, label_data, img_dir):
    """Handles trial execution with standardized feedback positioning."""
    has_img, img_path, target, choices = setup_trial_visuals(trial, components, label_data, img_dir, False)
    
    try:
        correct_key = ['1', '2', '9', '0'][choices.index(target)]
    except ValueError:
        correct_key = None

    is_slow = (trial.get('speed') == 'slow')
    is_repeat = trial.get('is_repeat', False)
    
    durations = {'img': 2.0 if is_slow else 1.0, 'isi1': 1.0 if is_slow else 0.5,
                 'dec': 10.0 if is_slow else 3.0, 'fb': 5.0 if is_slow else 1.5, 'iti': 1.5 if is_slow else 2.0}

    t0 = clock.getTime()
    
    # 1. Encoding
    while clock.getTime() < (t0 + durations['img'] + durations['isi1']):
        draw_extras(components, has_img, False)
        win.flip()

    # 2. Decision
    t_dec = clock.getTime()
    response, rt = None, None
    while clock.getTime() < (t_dec + durations['dec']):
        draw_extras(components, has_img, False)
        if is_slow:
            components['prompt'].draw()
            for lbl in components['key_labels']: lbl.draw()
        
        if response is None:
            response, rt = check_response(components, clock, t_dec)
            if response: break
        
        draw_buttons(components['buttons'], response)
        win.flip()

    # 3. Feedback Phase
    t_fb = clock.getTime()
    is_correct = (response == correct_key)
    components['fb_main'].text = ""
    components['fb_hint'].text = ""
    
    if is_slow:
        if is_correct:
            components['fb_main'].text = "The green highlight indicates that your choice was correct."
            components['fb_main'].color = 'green'
        else:
            components['fb_main'].text = "The red highlight indicates that your choice was incorrect."
            components['fb_main'].color = 'red'
            components['fb_hint'].text = "The correct option is highlighted in green."
    elif not is_correct and not is_slow and is_repeat:
        components['fb_main'].text = "Check the feedback and try again"
        components['fb_main'].color = 'red'
        components['fb_hint'].text = "The correct option is highlighted in green."

    while clock.getTime() < (t_fb + durations['fb']):
        draw_extras(components, has_img, False)
        draw_buttons_feedback(components['buttons'], response, correct_key)
        if response is None: components['respond_text'].draw()
        components['fb_main'].draw()
        components['fb_hint'].draw()
        win.flip()

    # 4. ITI
    t_iti = clock.getTime()
    while clock.getTime() < (t_iti + durations['iti']):
        components['fixation'].draw()
        win.flip()

    return {'accuracy': 1 if is_correct else 0}

def run_practice():
    win, components = create_window_and_components(demo_mode=False)
    
    # Initialize components using standardized layout variables
    components['prompt'] = visual.TextStim(win, text="Choose an option by pressing a key", pos=(0, Y_PROMPT), height=0.035, color='blue')
    components['fb_main'] = visual.TextStim(win, text="", pos=(0, Y_FEEDBACK_MAIN), height=0.03, wrapWidth=0.8)
    components['fb_hint'] = visual.TextStim(win, text="", pos=(0, Y_FEEDBACK_HINT), height=0.03, wrapWidth=0.8, color='green')
    
    components['key_labels'] = []
    for i, key in enumerate(['1', '2', '9', '0']):
        x = components['buttons'][i]['box'].pos[0]
        components['key_labels'].append(visual.TextStim(win, text=key, pos=(x, Y_BUTTON_LABELS), height=0.025, color='black'))

    label_map, _ = load_label_map(PRACTICE_LABEL_CSV, 'ObjectSpace')
    label_pool = build_participant_label_pool(pd.DataFrame(PRACTICE_DESIGN), label_map)
    label_data = (label_map, label_pool)

    # 1. Instructions
    show_instruction_screen(win, "PRACTICE TASK\n\nThis is a practice version of the task you will complete in the MRI scanner.\n\n\
The aim is to learn the correct names for different objects. The names for each object won't change during the experiment.\
\n\nYou will first see an object on the screen. After a delay, options will appear under the image.\
Your task is to choose the option you think is correct. This will be followed by feedback and the correct option will be shown if you were incorrect in your choice.")

    show_instruction_screen(win, "SLOW START\n\nThe first few trials will be slow.\n\nPrompts on screen will guide you. \n\nPlace your fingers like this onto the '1', '2', '9', and '0' keys", image_path="instruction_image_1.png")

    # 2. Trial Loop
    clock = core.Clock()
    for i, trial in enumerate(PRACTICE_DESIGN):
        if trial['speed'] == 'fast' and PRACTICE_DESIGN[i-1]['speed'] == 'slow':
            show_instruction_screen(win, "Well done!\n\nTrials will now move faster at the actual experimental speed.\nGuidance prompts will also be removed.\n\nTry your best to make correct choices.")

        repeat_enabled = (trial['speed'] == 'fast' and trial['is_repeat'])
        success = False
        while not success:
            result = run_custom_practice_trial(win, clock, dict(trial, image_file=f"{trial['ObjectSpace']}.png"), components, label_data, PRACTICE_IMG_DIR)
            if result['accuracy'] == 1 or not repeat_enabled:
                success = True

    show_instruction_screen(win, "PRACTICE COMPLETE.\n\nPlease inform the experimenter that you are done.")
    win.close()

if __name__ == "__main__":
    run_practice()