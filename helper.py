# Mostly from https://github.com/patrickloeber/snake-ai-pytorch/blob/main/helper.py

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def save_plot(save_path):
    plt.savefig(save_path, format="png")  # Save as PNG
    print(f"Plot saved as '{save_path}'.")

def choose_action(model, state):
  q_values = model.predict(state[np.newaxis, :], verbose=0)
  return np.argmax(q_values[0])