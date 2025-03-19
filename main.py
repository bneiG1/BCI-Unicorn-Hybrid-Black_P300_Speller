import tkinter as tk
from app.speller import update_layout, add_to_selected, stop_flashing, end_cooldown
from network.udp import send_character as send_udp_character
from network.tcp import send_character as send_tcp_character
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the configuration
with open("./config/config.json", "r") as file:
    config = json.load(file)

udp_ip = config["network"]["udp"]["ip"]
udp_port = config["network"]["udp"]["port"]

tcp_ip = config["network"]["tcp"]["ip"]
tcp_port = config["network"]["tcp"]["port"]

boards = config["boards"]
timings = config["timings"]

qwerty_6x6_order = boards[0]["qwerty_6x6"]
alphabetical_6x6_order = boards[1]["alphabetical_6x6"]
qwerty_order = boards[2]["qwerty"]
alphabetical_order = boards[3]["alphabetical"]

cooldown_time = timings["cooldown"]
times_flashing = timings["times_flashing"]

# Create the main window
root = tk.Tk()

# Set the title of the window
root.title("P300 Speller")

# Set the size of the window (width x height)
root.geometry("800x600")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a menu bar
menu_bar = tk.Menu(root)

# Create a dropdown menu to select the network protocol
protocol_menu = tk.Menu(menu_bar, tearoff=0)
protocol_var = tk.StringVar(value="UDP")
protocol_menu.add_radiobutton(label="UDP", variable=protocol_var, value="UDP")
protocol_menu.add_radiobutton(label="TCP", variable=protocol_var, value="TCP")
menu_bar.add_cascade(label="Protocol", menu=protocol_menu)

# Create a dropdown menu to select the layout
layout_menu = tk.Menu(menu_bar, tearoff=0)
layout_var = tk.StringVar(value="QWERTY")
layout_menu.add_radiobutton(
    label="QWERTY",
    variable=layout_var,
    value="QWERTY",
    command=lambda: update_layout(
        qwerty_order,
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
layout_menu.add_radiobutton(
    label="Alphabetical",
    variable=layout_var,
    value="Alphabetical",
    command=lambda: update_layout(
        alphabetical_order,
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
layout_menu.add_radiobutton(
    label="QWERTY 6x6",
    variable=layout_var,
    value="QWERTY 6x6",
    command=lambda: update_layout(
        qwerty_6x6_order,
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
layout_menu.add_radiobutton(
    label="Alphabetical 6x6",
    variable=layout_var,
    value="Alphabetical 6x6",
    command=lambda: update_layout(
        alphabetical_6x6_order,
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
menu_bar.add_cascade(label="Layout", menu=layout_menu)

# Create a dropdown menu to select the flashing mode
flashing_menu = tk.Menu(menu_bar, tearoff=0)
flashing_var = tk.StringVar(value="Random")
flashing_menu.add_radiobutton(
    label="Random",
    variable=flashing_var,
    value="Random",
    command=lambda: update_layout(
        (
            qwerty_order
            if layout_var.get() == "QWERTY"
            else (
                alphabetical_order
                if layout_var.get() == "Alphabetical"
                else (
                    qwerty_6x6_order
                    if layout_var.get() == "QWERTY 6x6"
                    else alphabetical_6x6_order
                )
            )
        ),
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
flashing_menu.add_radiobutton(
    label="Row/Column",
    variable=flashing_var,
    value="Row/Column",
    command=lambda: update_layout(
        (
            qwerty_order
            if layout_var.get() == "QWERTY"
            else (
                alphabetical_order
                if layout_var.get() == "Alphabetical"
                else (
                    qwerty_6x6_order
                    if layout_var.get() == "QWERTY 6x6"
                    else alphabetical_6x6_order
                )
            )
        ),
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
menu_bar.add_cascade(label="Flashing Mode", menu=flashing_menu)

# Add the menu bar to the root window
root.config(menu=menu_bar)

# Create a frame to hold the table
frame = tk.Frame(root)
frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Center the frame in the middle of the window
frame.place(relx=0.5, rely=0.35, anchor="center")

# Create a frame for control buttons
control_frame = tk.Frame(root)
control_frame.grid(row=1, column=0, pady=10, sticky="n")

# Create Start button
start_button = tk.Button(
    control_frame,
    text="Start",
    font=("Courier", 20),
    command=lambda: update_layout(
        (
            qwerty_order
            if layout_var.get() == "QWERTY"
            else (
                alphabetical_order
                if layout_var.get() == "Alphabetical"
                else (
                    qwerty_6x6_order
                    if layout_var.get() == "QWERTY 6x6"
                    else alphabetical_6x6_order
                )
            )
        ),
        frame,
        layout_var,
        flashing_var,
        selected_text,
        root,
        cooldown_time,
        times_flashing,
        udp_ip,
        udp_port,
        tcp_ip,
        tcp_port,
        protocol_var.get(),
    ),
)
start_button.pack(side=tk.LEFT, padx=20)

# Create Stop button
stop_button = tk.Button(
    control_frame,
    text="Stop",
    font=("Courier", 20),
    command=lambda: stop_flashing(frame),
)
stop_button.pack(side=tk.LEFT, padx=20)

# Create a text widget to display selected letters
selected_text = tk.Text(root, height=1, font=("Courier", 50), state=tk.DISABLED)
selected_text.grid(row=2, column=0, pady=10, sticky="ew")

# Set the initial layout
update_layout(
    qwerty_order,
    frame,
    layout_var,
    flashing_var,
    selected_text,
    root,
    cooldown_time,
    times_flashing,
    udp_ip,
    udp_port,
    tcp_ip,
    tcp_port,
    protocol_var.get(),
)

# Run the application
root.mainloop()
