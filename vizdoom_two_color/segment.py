#!/usr/bin/env python3

#####################################################################
# This script presents labels buffer that shows only visible game objects
# (enemies, pickups, exploding barrels etc.), each with unique label.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
from argparse import ArgumentParser
from random import choice

import cv2
import numpy as np

import vizdoom as vzd
from typing import Dict
from collections import defaultdict

DEFAULT_CONFIG = os.path.join(os.getcwd(), "scenarios/k_item.cfg")

if __name__ == "__main__":
    parser = ArgumentParser(
        "ViZDoom example showing how to use labels and labels buffer."
    )
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    game.set_render_hud(False)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # Enables labeling of the in game objects.
    game.set_labels_buffer_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.init()

    actions = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]

    episodes = 10

    # Sleep time between actions in ms
    sleep_time = 28

    # Prepare some colors and drawing function
    # Colors in in BGR order (expected by OpenCV)
    doom_red_color = [0, 0, 203]
    doom_blue_color = [203, 0, 0]

    def draw_bounding_box(buffer, x, y, width, height, color):
        """
        Draw a rectangle (bounding box) on a given buffer in the given color.
        """
        for i in range(width):
            buffer[y, x + i, :] = color
            buffer[y + height, x + i, :] = color

        for i in range(height):
            buffer[y + i, x, :] = color
            buffer[y + i, x + width, :] = color
    from matplotlib import cm

    CM = cm.jet        

    DEFAULT_LABELS_DEF = defaultdict(lambda : 0)
    DEFAULT_LABELS_DEF["Floor/Ceil"] = 240
    DEFAULT_LABELS_DEF["Wall"] = 210
    DEFAULT_LABELS_DEF["ItemFog"] = 40
    DEFAULT_LABELS_DEF["TeleportFog"] = 60
    DEFAULT_LABELS_DEF["BulletPuff"] = 80
    DEFAULT_LABELS_DEF["Blood"] = 100
    DEFAULT_LABELS_DEF["TallGreenColumn"] = 114
    DEFAULT_LABELS_DEF["TallRedColumn"] = 20
    DEFAULT_LABELS_DEF["BlueArmor"] = 20
    DEFAULT_LABELS_DEF["GreenArmor"] = 114
    DEFAULT_LABELS_DEF["DeadDoomPlayer"] = 200
    DEFAULT_LABELS_DEF["DoomPlayer"] = 220
    DEFAULT_LABELS_DEF["Self"] = 240

    DEFAULT_LABELS_DEF_RGB = defaultdict(lambda : np.asfarray([CM(0.)[:3]]).T.copy())
    for obj_name, obj_val in DEFAULT_LABELS_DEF.items():
        DEFAULT_LABELS_DEF_RGB[obj_name] = np.asfarray([CM(obj_val / 240.)[:3]]).T.copy()

    def semseg_rgb(state: vzd.GameState, label_def: Dict[str, int]=None) -> np.ndarray:
        if label_def is None:
            label_def = DEFAULT_LABELS_DEF_RGB
        raw_buffer: np.ndarray = state.labels_buffer
        buffer:     np.ndarray = np.empty((3, *raw_buffer.shape), dtype=float)
        buffer[:, :, :] = label_def[''].reshape((3, 1, 1))
        buffer[:, raw_buffer == 1] = label_def["Wall"]
        buffer[:, raw_buffer == 0] = label_def["Floor/Ceil"]


        if state.labels and "Self" in label_def:
            for obj in state.labels[:-1]:
                buffer[:, raw_buffer == obj.value] = label_def[obj.object_name]
            
            last_obj = state.labels[-1]
            if last_obj.object_name == "DoomPlayer":
                buffer[:, raw_buffer == last_obj.value] = label_def["Self"]
            else:
                buffer[:, raw_buffer == last_obj.value] = label_def[last_obj.object_name]
        else:
            for obj in state.labels:
                buffer[:, raw_buffer == obj.value] = label_def[obj.object_name]
        
        return buffer        

    def color_labels(labels):
        """
        Walls are blue, floor/ceiling are red (OpenCV uses BGR).
        """
        tmp = np.stack([labels] * 3, -1)
        tmp[labels == 0] = [255, 0, 0]
        tmp[labels == 1] = [0, 0, 255]
        tmp[labels == 24] = [0, 0, 255]        
        tmp[labels == 17] = [102, 155, 155]
        tmp[labels == 34] = [0, 102, 51]
        tmp[labels == 51] = [51, 255, 51]
        tmp[labels == 68] = [153, 76, 0]
        tmp[labels == 85] = [255, 178, 102]
        tmp[labels == 102] = [0, 102, 51]
        tmp[labels == 119] = [255, 255, 0]
        tmp[labels == 136] = [255, 255, 0]
        tmp[labels == 153] = [0, 51, 51]
        tmp[labels == 170] = [215, 41, 51]
        tmp[labels == 187] = [115, 141, 51]
        tmp[labels == 204] = [165, 41, 151]
        tmp[labels == 221] = [115, 241, 11]
        tmp[labels == 238] = [215, 241, 11]
        tmp[labels == 229] = [215, 241, 11]
        tmp[labels == 178] = [215, 41, 51]
        tmp[labels == 204] = [215, 141, 141]
        return tmp

    for i in range(episodes):
        print(f"Episode #{i + 1}")

        seen_in_this_episode = set()

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Get the state
            state = game.get_state()

            # Get labels buffer, that is always in 8-bit grey channel format.
            # Show only visible game objects (enemies, pickups, exploding barrels etc.), each with a unique label.
            # Additional labels data are available in state.labels.
            labels = state.labels_buffer
            if labels is not None:
                cv2.imshow("ViZDoom Labels Buffer", semseg_rgb(state).transpose(1, 2, 0))
                cv2.waitKey(sleep_time)
            #if labels is not None:
            #    cv2.imshow("ViZDoom Labels Buffer", color_labels(labels))
            #    cv2.waitKey(sleep_time)

            # Get screen buffer, given in selected format. This buffer is always available.
            # Using information from state.labels draw bounding boxes.
            screen = state.screen_buffer
            for label in state.labels:
                print(f"{label.object_name}:{label.object_id}:{label.value}")
                if label.object_name in ["Medkit", "GreenArmor"]:
                    draw_bounding_box(
                        screen,
                        label.x,
                        label.y,
                        label.width,
                        label.height,
                        doom_blue_color,
                    )
                else:
                    draw_bounding_box(
                        screen,
                        label.x,
                        label.y,
                        label.width,
                        label.height,
                        doom_red_color,
                    )
            cv2.imshow("ViZDoom Screen Buffer", screen)

            cv2.waitKey(sleep_time)

            # Make random action
            game.make_action(choice(actions))

            print(f"State #{state.number}")
            print(
                "Player position: x:",
                state.game_variables[0],
                ", y:",
                state.game_variables[1],
                ", z:",
                state.game_variables[2],
            )
            print("Labels:")

            # Print information about objects visible on the screen.
            # object_id identifies a specific in-game object.
            # It's unique for each object instance (two objects of the same type on the screen will have two different ids).
            # object_name contains the name of the object (can be understood as type of the object).
            # value tells which value represents the object in labels_buffer.
            # Values decrease with the distance from the player.
            # Objects with higher values (closer ones) can obscure objects with lower values (further ones).
            for label in state.labels:
                seen_in_this_episode.add(label.object_name)
                # print("---------------------")
                print(
                    "Label:",
                    label.value,
                    ", object id:",
                    label.object_id,
                    ", object name:",
                    label.object_name,
                )
                print(
                    "Object position: x:",
                    label.object_position_x,
                    ", y:",
                    label.object_position_y,
                    ", z:",
                    label.object_position_z,
                )

                # Other available fields (position and velocity and bounding box):
                # print("Object rotation angle", label.object_angle, "pitch:", label.object_pitch, "roll:", label.object_roll)
                # print("Object velocity x:", label.object_velocity_x, "y:", label.object_velocity_y, "z:", label.object_velocity_z)
                print(
                    "Bounding box: x:",
                    label.x,
                    ", y:",
                    label.y,
                    ", width:",
                    label.width,
                    ", height:",
                    label.height,
                )

            print("=====================")

        print("Episode finished!")

        print("=====================")

        print("Unique objects types seen in this episode:")
        for label in seen_in_this_episode:
            print(label)

        print("************************")

    cv2.destroyAllWindows()