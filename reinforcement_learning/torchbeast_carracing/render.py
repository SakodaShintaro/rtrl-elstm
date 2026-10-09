# Real-time view of the agent, in the style of vla_streaming_rl's render strip:
# the full environment view next to the observation the network reads, with
# the scalars of this tick listed below it.

import cv2
import numpy as np

# Discrete CarRacing-v3 actions (gymnasium car_racing.py).
ACTION_NAMES = ["noop", "right", "left", "gas", "brake"]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.4
_LINE_HEIGHT = 16


def append_rows(image, rows):
    """Add a dark band listing (name, value) rows under the image."""
    band = np.full((_LINE_HEIGHT * len(rows) + 6, image.shape[1], 3), 50, dtype=np.uint8)
    value_x = image.shape[1] // 2
    for i, (name, value) in enumerate(rows):
        y = (i + 1) * _LINE_HEIGHT
        cv2.putText(band, name, (5, y), _FONT, _FONT_SCALE, (190, 190, 190), 1)
        cv2.putText(band, value, (value_x, y), _FONT, _FONT_SCALE, (255, 255, 255), 1)
    return np.vstack((image, band))


def add_label(image, label):
    band = np.full((20, image.shape[1], 3), 0, dtype=np.uint8)
    cv2.putText(band, label, (5, 14), _FONT, 0.5, (255, 255, 255), 1)
    return np.vstack((band, image))


def concat_panels(panels):
    """Lay out named RGB panels side by side, padded to a common height."""
    labeled = [add_label(image, label) for label, image in panels.items()]
    height = max(image.shape[0] for image in labeled)
    padded = [
        np.vstack((image, np.zeros((height - image.shape[0], image.shape[1], 3), np.uint8)))
        for image in labeled
    ]
    return np.hstack(padded)


def render_frame(env_image, obs, scale, action, probs, value, reward, episode_return,
                 episode_step, global_step):
    """One BGR frame. `obs` is the (C, H, W) uint8 observation the network reads
    next; `action`, `probs` and `value` are from the tick that led to it."""
    obs_viz = obs.transpose(1, 2, 0)
    h, w = obs_viz.shape[:2]
    obs_viz = cv2.resize(obs_viz, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    rows = [("action", ACTION_NAMES[action])]
    rows += [(f"  p({name})", f"{p:.3f}") for name, p in zip(ACTION_NAMES, probs)]
    rows += [("entropy", f"{-sum(p * np.log(max(p, 1e-12)) for p in probs):.3f}")]
    rows += [
        ("value", f"{value:+.3f}"),
        ("reward", f"{reward:+.3f}"),
        ("episode_return", f"{episode_return:+.2f}"),
        ("episode_step", f"{episode_step}"),
        ("global_step", f"{global_step}"),
    ]
    panels = {
        "environment": env_image,
        "observation": append_rows(obs_viz, rows),
    }
    return cv2.cvtColor(concat_panels(panels), cv2.COLOR_RGB2BGR)
