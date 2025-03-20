import tkinter as tk
from interface.speller import update_layout, stop_flashing
import logging
from config.config import  tcp_port, tcp_ip, udp_sender_port, udp_sender_ip, times_flashing, cooldown, qwerty, alphabetical, qwerty_6x6, alphabetical_6x6, source  # Import source configuration
# Import acquisition controls
from network.UnicornPy_acquisition import start_acquisition, stop_acquisition
from network.udp_listener import listen_for_character  # Import UDP listener
import threading  # For running UDP listener in a separate thread

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper function to get the current layout order


def get_layout_order(layout):
    if layout == "QWERTY":
        return qwerty
    elif layout == "Alphabetical":
        return alphabetical
    elif layout == "QWERTY 6x6":
        return qwerty_6x6
    elif layout == "Alphabetical 6x6":
        return alphabetical_6x6

# Helper function to update layout with current settings


def update_initial_layout():
    update_layout(
        get_layout_order(layout_var.get()),
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown,
        times_flashing,
        udp_sender_ip,
        udp_sender_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
        False
    )


def update_current_layout():
    update_layout(
        get_layout_order(layout_var.get()),
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown,
        times_flashing,
        udp_sender_ip,
        udp_sender_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    )


# Create the main window
root = tk.Tk()
root.title("P300 Speller")
root.geometry("800x700")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a menu bar
menu_bar = tk.Menu(root)

# Protocol menu
protocol_menu = tk.Menu(menu_bar, tearoff=0)
protocol_var = tk.StringVar(value="UDP")
protocol_menu.add_radiobutton(label="UDP", variable=protocol_var, value="UDP")
protocol_menu.add_radiobutton(label="TCP", variable=protocol_var, value="TCP")
menu_bar.add_cascade(label="Protocol", menu=protocol_menu)

# Layout menu
layout_menu = tk.Menu(menu_bar, tearoff=0)
layout_var = tk.StringVar(value="QWERTY")
for layout in ["QWERTY", "Alphabetical", "QWERTY 6x6", "Alphabetical 6x6"]:
    layout_menu.add_radiobutton(
        label=layout,
        variable=layout_var,
        value=layout,
        command=update_current_layout,
    )
menu_bar.add_cascade(label="Layout", menu=layout_menu)

# Flashing mode menu
flashing_menu = tk.Menu(menu_bar, tearoff=0)
flashing_var = tk.StringVar(value="Random")
for mode in ["Random", "Row/Column"]:
    flashing_menu.add_radiobutton(
        label=mode,
        variable=flashing_var,
        value=mode,
        command=update_current_layout,
    )
menu_bar.add_cascade(label="Flashing Mode", menu=flashing_menu)

# Add the menu bar to the root window
root.config(menu=menu_bar)

# Create a frame to hold the table
frame = tk.Frame(root)
frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.place(relx=0.5, rely=0.35, anchor="center")

# Create a frame for control buttons
control_frame = tk.Frame(root)
control_frame.grid(row=1, column=0, pady=10, sticky="n")

# Start button
start_button = tk.Button(
    control_frame,
    text="Start",
    font=("Courier", 20),
    # Start data source and update layout
    command=lambda: [start_acquisition(), update_current_layout()],
)
start_button.pack(side=tk.LEFT, padx=20)

# Stop button
stop_button = tk.Button(
    control_frame,
    text="Stop",
    font=("Courier", 20),
    # Stop flashing and pause acquisition
    command=lambda: [stop_flashing(frame), stop_acquisition()],
)
stop_button.pack(side=tk.LEFT, padx=20)

# Text widget to display selected letters
selected_text = tk.Text(root, height=1, font=(
    "Courier", 50), state=tk.DISABLED)
selected_text.grid(row=2, column=0, pady=10, sticky="ew")

# Set the initial layout
update_initial_layout()

# Run the application
root.mainloop()
