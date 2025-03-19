import tkinter as tk
from utils.utils import create_flashing_overlay, create_row_col_flashing, show_overlay
from network.udp import send_character as send_udp_character
from network.tcp import send_character as send_tcp_character
import logging  # Add this import

cooldown = False  # Global variable to manage cooldown state
flashing_complete = False  # Global variable to manage flashing state


# Function to update the keyboard layout
def update_layout(
    order,
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
    protocol,
):
    global cooldown, flashing_complete
    logging.info(
        f"Updating layout to {layout_var.get()} with flashing mode {flashing_var.get()} and protocol {protocol}"
    )
    cooldown = False
    flashing_complete = False
    for widget in frame.winfo_children():
        widget.destroy()
    labels = []
    for row_index, row in enumerate(order):
        row_labels = []
        for col_index, letter in enumerate(row):
            label = tk.Label(
                frame,
                text=letter,
                font=("Courier", 50),
                padx=10,
                pady=5,
                anchor="center",
                justify="center",
            )
            label.grid(row=row_index, column=col_index, sticky="nsew")
            label.bind(
                "<Button-1>",
                lambda e, l=letter: add_to_selected(
                    l,
                    selected_text,
                    root,
                    frame,
                    layout_var,
                    flashing_var,
                    order,
                    cooldown_time,
                    times_flashing,
                    udp_ip,
                    udp_port,
                    tcp_ip,
                    tcp_port,
                    protocol,
                ),
            )
            row_labels.append(label)
        labels.append(row_labels)
        frame.grid_rowconfigure(row_index, weight=1)
    for col_index in range(len(order[0])):
        frame.grid_columnconfigure(col_index, weight=1)

    if flashing_var.get() == "Random":
        for row in labels:
            for label in row:
                create_flashing_overlay(
                    label,
                    cooldown,
                    times_flashing,
                    on_complete=lambda: set_flashing_complete(),
                )
    else:
        if len(order) == 6 and len(order[0]) == 6:
            create_row_col_flashing(
                labels,
                cooldown,
                root,
                times_flashing,
                on_complete=lambda: set_flashing_complete(),
            )


def set_flashing_complete():
    global flashing_complete
    flashing_complete = True
    logging.info("Flashing complete")


# Function to add selected letter to the display
def add_to_selected(
    letter,
    selected_text,
    root,
    frame,
    layout_var,
    flashing_var,
    order,
    cooldown_time,
    times_flashing,
    udp_ip,
    udp_port,
    tcp_ip,
    tcp_port,
    protocol,
):
    global cooldown, flashing_complete
    if not flashing_complete:
        logging.warning("Attempted to select a letter before flashing was complete")
        return
    logging.info(f"Selected letter: {letter}")
    cooldown = True
    selected_text.config(state=tk.NORMAL)
    selected_text.insert(tk.END, letter)
    selected_text.config(state=tk.DISABLED)
    stop_flashing(frame)
    show_overlay(letter, root, cooldown_time)
    if protocol == "UDP":
        send_udp_character(letter, udp_ip, udp_port)
    else:
        send_tcp_character(letter, tcp_ip, tcp_port)
    root.after(
        cooldown_time,
        lambda: end_cooldown(
            frame,
            layout_var,
            flashing_var,
            selected_text,
            root,
            order,
            cooldown_time,
            times_flashing,
            udp_ip,
            udp_port,
            tcp_ip,
            tcp_port,
            protocol,
        ),
    )


# Function to stop all flashing
def stop_flashing(frame):
    logging.info("Stopping all flashing")
    for widget in frame.winfo_children():
        if hasattr(widget, "flash_id"):
            widget.after_cancel(widget.flash_id)
        widget.config(background="white", foreground="black")


# Function to end the cooldown period
def end_cooldown(
    frame,
    layout_var,
    flashing_var,
    selected_text,
    root,
    order,
    cooldown_time,
    times_flashing,
    udp_ip,
    udp_port,
    tcp_ip,
    tcp_port,
    protocol,
):
    global cooldown
    logging.info("Cooldown period ended")
    cooldown = False
    update_layout(
        order,
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
        protocol,
    )
