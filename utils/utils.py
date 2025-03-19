import tkinter as tk
import random
import logging  # Add this import

# Function to create a flashing overlay on a label
def create_flashing_overlay(label, cooldown, times_flashing, on_complete):
    global flashing_complete
    flash_count = 0
    def flash():
        nonlocal flash_count
        if not cooldown and flash_count < times_flashing:
            current_color = label.cget("background")
            next_color = "black" if current_color == "white" else "white"
            label.config(
                background=next_color,
                foreground="white" if next_color == "black" else "black",
            )
            flash_duration = random.randint(60, 125)
            inter_stimulus_interval = random.randint(110, 125)
            if hasattr(label, 'flash_id'):
                label.after_cancel(label.flash_id)
            label.flash_id = label.after(
                flash_duration, lambda: label.config(background="white", foreground="black")
            )
            label.flash_id = label.after(flash_duration + inter_stimulus_interval, flash)
            flash_count += 1
        else:
            flashing_complete = True
            logging.info("Flashing overlay complete")
            on_complete()
    flash()

# Function to create row/column flashing overlay
def create_row_col_flashing(labels, cooldown, root, times_flashing, on_complete):
    global flashing_complete
    flash_count = 0
    def flash_row_col():
        nonlocal flash_count
        if not cooldown and flash_count < times_flashing:
            row_index = random.randint(0, len(labels) - 1)
            col_index = random.randint(0, len(labels[0]) - 1)
            
            row_labels = labels[row_index]
            col_labels = [labels[r][col_index] for r in range(len(labels))]
            
            for label in row_labels + col_labels:
                label.config(background="black", foreground="white")
            
            flash_duration = random.randint(60, 125)
            inter_stimulus_interval = random.randint(110, 125)
            
            root.after(
                flash_duration,
                lambda: [label.config(background="white", foreground="black") for label in row_labels + col_labels]
            )
            root.after(flash_duration + inter_stimulus_interval, flash_row_col)
            flash_count += 1
        else:
            flashing_complete = True
            logging.info("Row/Column flashing complete")
            on_complete()
    flash_row_col()

# Function to show overlay with selected letter
def show_overlay(letter, root, cooldown_time):
    logging.info(f"Showing overlay for letter: {letter}")
    overlay_frame = tk.Frame(root, background="white")
    overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
    overlay_label = tk.Label(
        overlay_frame,
        text=letter,
        font=("Courier", 200),
        padx=10,
        pady=5,
        anchor="center",
        justify="center",
        background="white"
    )
    overlay_label.pack(expand=True)
    root.after(cooldown_time, overlay_frame.destroy)
