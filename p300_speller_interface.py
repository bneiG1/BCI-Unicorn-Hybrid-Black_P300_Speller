"""
P300 Speller Interface
======================
This module provides a visual interface for the P300 speller using pygame.
It displays a matrix of characters and manages the flashing stimuli.
"""

from matplotlib.pylab import matrix
import pygame
import random
import time
import threading
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

# Import our Unicorn connection module
from unicorn_p300_speller import UnicornP300Speller
from p300_classifier import P300Classifier, ClassifierConfig
from calibration_handler import CalibrationHandler, CalibrationConfig
from eeg_preprocessing import EEGPreprocessor, PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P300SpellerInterface")

# Default colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Color schemes
COLOR_SCHEMES = {
    "default": {
        "background": BLACK,
        "cell": GRAY,
        "text": BLACK,
        "highlight": WHITE,
        "result_text": YELLOW,
    },
    "high_contrast": {
        "background": BLACK,
        "cell": BLUE,
        "text": WHITE,
        "highlight": YELLOW,
        "result_text": GREEN,
    },
    "light": {
        "background": WHITE,
        "cell": CYAN,
        "text": BLACK,
        "highlight": RED,
        "result_text": BLUE,
    },
    "dark": {
        "background": BLACK,
        "cell": (50, 50, 50),
        "text": WHITE,
        "highlight": (200, 200, 200),
        "result_text": GREEN,
    },
}


class P300SpellerState(Enum):
    """States of the P300 speller."""

    IDLE = "idle"  # Waiting to start
    TRAINING = "training"  # Collecting training data
    CLASSIFYING = "classifying"  # Normal operation
    ADAPTING = "adapting"  # Online adaptation phase


class P300SpellerInterface:
    """
    Visual interface for the P300 speller using pygame.
    Displays a matrix of characters and manages the flashing stimuli.
    """

    # Default timing parameters
    FLASH_DURATION = 125  # milliseconds
    INTER_STIMULUS_INTERVAL = 125  # milliseconds
    SEQUENCES_PER_TRIAL = 10  # Number of sequences per character selection

    # Default P300 speller matrix (6x6)
    DEFAULT_MATRIX = [
        ["A", "B", "C", "D", "E", "F"],
        ["G", "H", "I", "J", "K", "L"],
        ["M", "N", "O", "P", "Q", "R"],
        ["S", "T", "U", "V", "W", "X"],
        ["Y", "Z", "0", "1", "2", "3"],
        ["4", "5", "6", "7", "8", "9"],
    ]

    # Alternative matrices that can be selected
    ALTERNATIVE_MATRICES = {
        "alpha_numeric": [
            ["A", "B", "C", "D", "E", "F"],
            ["G", "H", "I", "J", "K", "L"],
            ["M", "N", "O", "P", "Q", "R"],
            ["S", "T", "U", "V", "W", "X"],
            ["Y", "Z", "0", "1", "2", "3"],
            ["4", "5", "6", "7", "8", "9"],
        ],
        "with_specials": [
            ["A", "B", "C", "D", "E", "F"],
            ["G", "H", "I", "J", "K", "L"],
            ["M", "N", "O", "P", "Q", "R"],
            ["S", "T", "U", "V", "W", "X"],
            ["Y", "Z", ".", ",", "?", "!"],
            ["0", "1", "2", "3", "_", " "],
        ],
        "qwerty": [
            ["1", "2", "3", "4", "5", "6"],
            ["Q", "W", "E", "R", "T", "Y"],
            ["A", "S", "D", "F", "G", "H"],
            ["Z", "X", "C", "V", "B", "N"],
            ["7", "8", "9", "0", "U", "I"],
            ["O", "P", "J", "K", "L", "_"],
        ],
    }

    def __init__(
        self,
        width=800,
        height=600,
        color_scheme="default",
        matrix="abc",
        flash_duration=None,
        isi=None,
        sequences=None,
        flash_brightness=0.5,
        eeg_device=None,
    ):
        """
        Initialize the P300 Speller Interface.

        Args:
            eeg_device: The UnicornP300Speller device for EEG acquisition
            matrix: Custom character matrix (default is 6x6 with letters and numbers)
            width: Window width
            height: Window height
            color_scheme: Name of the color scheme to use (from COLOR_SCHEMES)
            flash_duration: Custom flash duration in ms (overrides default)
            isi: Custom inter-stimulus interval in ms (overrides default)
            sequences: Custom number of sequences per trial (overrides default)
            flash_brightness: Flash brightness multiplier (0.0-1.0)
        """
        self.eeg_device = eeg_device
        self.matrix_type = "default"

        # Set matrix based on input or use default
        if isinstance(matrix, str) and matrix in self.ALTERNATIVE_MATRICES:
            self.matrix = self.ALTERNATIVE_MATRICES[matrix]
            self.matrix_type = matrix
        elif matrix:
            self.matrix = matrix
        else:
            self.matrix = self.DEFAULT_MATRIX

        # Set display parameters
        self.width = width
        self.height = height

        # Set color scheme
        self.color_scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["default"])

        # Set flash parameters
        self.flash_duration = (
            flash_duration if flash_duration is not None else self.FLASH_DURATION
        )
        self.isi = isi if isi is not None else self.INTER_STIMULUS_INTERVAL
        self.sequences_per_trial = (
            sequences if sequences is not None else self.SEQUENCES_PER_TRIAL
        )
        self.flash_brightness = min(max(flash_brightness, 0.0), 1.0)  # Clamp to 0.0-1.0

        # Screen and pygame-related attributes
        self.screen = None
        self.font = None
        self.small_font = None
        self.clock = None
        self.running = False

        # Stimulus-related attributes
        self.current_flash = None  # Currently flashing row or column
        self.flash_start_time = 0  # When the current flash started
        self.last_flash_time = 0  # Last flash time for timing control
        self.is_flashing = False
        self.stimulus_thread = None

        # Results and text entry
        self.text_result = ""
        self.current_text = ""  # Current text being typed
        self.target_text = None  # Target text for training/evaluation
        self.target_letter = None  # Current target letter (for training/calibration)
        self.status_message = ""

        # Calculate dimensions
        self.rows = len(self.matrix)
        self.cols = len(self.matrix[0]) if self.rows > 0 else 0

        # Make responsive based on screen size
        self.recalculate_dimensions()

        # P300 classification components
        self.classifier = None
        if self.eeg_device:
            self.classifier = P300Classifier(
                sampling_rate=self.eeg_device.sampling_rate
            )
        # Initialize preprocessor and calibration handler
        if self.eeg_device:
            self.preprocessor = EEGPreprocessor(
                sampling_rate=self.eeg_device.sampling_rate,
                num_channels=len(self.eeg_device.CHANNEL_NAMES),
                channel_names=self.eeg_device.CHANNEL_NAMES,
                config=PreprocessingConfig,
            )

            self.calibration_handler = CalibrationHandler(
                sampling_rate=self.eeg_device.sampling_rate,
                config=CalibrationConfig,
                preprocessor=self.preprocessor,
                classifier=self.classifier,
            )
        else:
            self.preprocessor = None
            self.calibration_handler = None

        # Training and classification data
        self.current_training_char_index = 0
        self.current_training_sequence = 0
        self.training_chars = []
        self.training_complete = False
        self.required_training_trials = 20  # Number of training trials needed
        self.current_training_trial = 0

        # Current state
        self.state = P300SpellerState.IDLE

        # Epoch collection
        self.current_row_epochs = []
        self.current_col_epochs = []
        self.current_sequence_count = 0

        # Classification results
        self.classification_confidence = 0.0
        self.min_confidence_threshold = 0.7

        # Error correction
        self.error_correction_enabled = True
        self.undo_stack = []  # Store previous characters for error correction

        # Dynamic stopping
        self.dynamic_stopping_enabled = True
        self.confidence_threshold = 0.8
        self.min_sequences = 3
        self.max_sequences = 10

        # Adaptive timing parameters
        self.adaptive_timing_enabled = True
        self.base_flash_duration = 125  # Base flash duration in ms
        self.base_isi = 75  # Base inter-stimulus interval in ms
        self.performance_window = []  # Store recent accuracy for adaptation
        self.performance_window_size = 10
        self.timing_adjustment_step = 10  # ms
        self.min_flash_duration = 50
        self.max_flash_duration = 200
        self.min_isi = 50
        self.max_isi = 150

        # Visual feedback parameters
        self.accuracy_history = []
        self.max_history_size = 20
        self.feedback_alpha = 128  # Transparency for feedback overlay

        # Performance metrics
        self.current_accuracy = 0.0
        self.selection_speed = 0.0  # Characters per minute
        self.last_selection_time = time.time()

        logger.info(
            f"Initialized P300 Speller Interface with {self.rows}x{self.cols} matrix"
        )

    def initialize_pygame(self) -> None:
        """Initialize pygame and create the display window."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("P300 Speller")
        self.font = pygame.font.SysFont("Arial", 36)
        self.small_font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()
        logger.info("Pygame initialized")

    def start(self) -> None:
        """Start the P300 speller interface."""
        if not self.eeg_device:
            logger.warning("Starting without EEG device - demo mode only")
        elif not self.eeg_device.is_connected:
            if not self.eeg_device.connect():
                logger.error("Failed to connect to EEG device")
                return

        self.initialize_pygame()
        self.running = True

        # Start the stimulus presentation thread
        self.stimulus_thread = threading.Thread(target=self.stimulus_presentation_loop)
        self.stimulus_thread.daemon = True
        self.stimulus_thread.start()

        # Main pygame loop
        self.main_loop()

    def stop(self) -> None:
        """Stop the P300 speller interface."""
        self.running = False
        if self.stimulus_thread:
            self.stimulus_thread.join(timeout=1.0)

        if self.eeg_device and self.eeg_device.is_connected:
            self.eeg_device.disconnect()

        pygame.quit()
        logger.info("P300 Speller Interface stopped")

    def draw_matrix(self) -> None:
        """Draw the speller matrix on the screen."""
        # Clear the screen with background color from the chosen color scheme
        self.screen.fill(self.color_scheme["background"])

        # Draw the matrix cells using calculated responsive positions
        for i in range(self.rows):
            for j in range(self.cols):
                # Determine cell color (highlight if flashing)
                color = self.color_scheme["cell"]

                # Calculate cell brightness for flashing
                flash_multiplier = 1.0

                # If row is flashing
                if self.is_flashing and self.current_flash == f"row_{i}":
                    # Use highlight color with adjusted brightness
                    color = self.color_scheme["highlight"]
                    # Apply brightness adjustment
                    flash_multiplier = self.flash_brightness

                # If column is flashing
                elif self.is_flashing and self.current_flash == f"col_{j}":
                    color = self.color_scheme["highlight"]
                    flash_multiplier = self.flash_brightness

                # Apply brightness adjustment
                actual_color = tuple(min(255, int(c * flash_multiplier)) for c in color)

                # Draw the cell
                rect = pygame.Rect(
                    self.matrix_x + j * self.cell_width,
                    self.matrix_y + i * self.cell_height,
                    self.cell_width,
                    self.cell_height,
                )
                pygame.draw.rect(self.screen, actual_color, rect)
                pygame.draw.rect(
                    self.screen, self.color_scheme["background"], rect, 1
                )  # Border

                # Highlight the target letter if in training/calibration mode
                is_target = False
                if self.target_letter and self.matrix[i][j] == self.target_letter:
                    is_target = True
                    # Draw target indicator (circle around the letter)
                    target_rect = rect.inflate(-10, -10)
                    pygame.draw.ellipse(self.screen, RED, target_rect, 2)

                # Draw the character
                char = self.matrix[i][j]
                text_color = self.color_scheme["text"]
                if is_target:
                    text_color = RED
                text = self.font.render(char, True, text_color)
                text_rect = text.get_rect(
                    center=(
                        self.matrix_x + j * self.cell_width + self.cell_width // 2,
                        self.matrix_y + i * self.cell_height + self.cell_height // 2,
                    )
                )
                self.screen.blit(text, text_rect)

        # Draw the spelled text display area
        result_text_box_height = 60
        result_text_box_width = int(self.width * 0.8)
        result_text_box_x = (self.width - result_text_box_width) // 2
        result_text_box_y = self.matrix_y + self.matrix_height + 30

        # Draw text display box background
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            pygame.Rect(
                result_text_box_x,
                result_text_box_y,
                result_text_box_width,
                result_text_box_height,
            ),
        )
        pygame.draw.rect(
            self.screen,
            GRAY,
            pygame.Rect(
                result_text_box_x,
                result_text_box_y,
                result_text_box_width,
                result_text_box_height,
            ),
            2,
        )

        # Draw the result text
        result_text = self.small_font.render(
            f"Text: {self.text_result}", True, self.color_scheme["result_text"]
        )
        result_rect = result_text.get_rect(
            center=(self.width // 2, result_text_box_y + result_text_box_height // 2)
        )
        self.screen.blit(result_text, result_rect)

        # Draw status information panel
        status_panel_y = 10
        status_panel_height = 90
        status_panel_width = 200

        # Draw semi-transparent panel background
        status_surface = pygame.Surface(
            (status_panel_width, status_panel_height), pygame.SRCALPHA
        )
        status_surface.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(status_surface, (10, status_panel_y))

        # Draw status info
        status_info = []
        if self.eeg_device and self.eeg_device.is_connected:
            status_info.append(f"Device: Connected")
        else:
            status_info.append(f"Device: Not connected")

        status_info.append(f"Matrix: {self.matrix_type}")
        status_info.append(f"Flash: {self.flash_duration}ms / ISI: {self.isi}ms")

        # Draw status text
        for i, status in enumerate(status_info):
            status_text = self.small_font.render(status, True, WHITE)
            self.screen.blit(status_text, (15, status_panel_y + 5 + i * 25))

        # Draw performance metrics
        self.draw_performance_feedback()

        # Update the display
        pygame.display.flip()

    def main_loop(self) -> None:
        """Main pygame event loop."""
        self.initialize_performance_tracking()
        running = True
        clock = pygame.time.Clock()

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Handle other key events...

            # Clear the screen
            self.screen.fill(BLACK)

            # Draw the matrix
            self.draw_matrix()

            # Handle stimulus presentation with adaptive timing
            if current_time - self.last_flash_time >= self.isi:
                self.flash_next_stimulus()
                self.last_flash_time = current_time

            # Process EEG data and update classifier
            if self.eeg_data_available():
                prediction = self.process_eeg_data()
                if prediction is not None:
                    correct = self.handle_prediction(prediction)
                    self.update_performance_metrics(correct)

            # Draw performance feedback
            self.draw_performance_feedback()

            # Update the display
            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        pygame.quit()

    def handle_prediction(self, predicted_char: str) -> bool:
        """Handle character prediction and return whether it was correct."""
        # For training/evaluation mode where we know the target
        is_correct = False
        if self.target_text and len(self.current_text) < len(self.target_text):
            target_char = self.target_text[len(self.current_text)]
            is_correct = predicted_char == target_char

        # Update text
        self.current_text += predicted_char

        return is_correct

    def stimulus_presentation_loop(self) -> None:
        """
        Background thread that manages the stimulus presentation.
        Runs series of row and column flashes according to the P300 speller paradigm,
        with precise timing control for EEG marker synchronization.
        """
        logger.info("Starting stimulus presentation loop")
        logger.info(
            f"Flash parameters: duration={self.flash_duration}ms, ISI={self.isi}ms"
        )

        while self.running:
            # Wait before starting a new sequence
            time.sleep(1.0)

            # Run sequences of row and column flashes
            for seq in range(self.sequences_per_trial):
                # Generate random order of rows and columns
                flash_order = []
                for i in range(self.rows):
                    flash_order.append(f"row_{i}")
                for j in range(self.cols):
                    flash_order.append(f"col_{j}")

                # Randomize the order
                random.shuffle(flash_order)

                # Execute each flash with precise timing control
                for flash_item in flash_order:
                    # Calculate timing parameters
                    flash_duration_sec = self.flash_duration / 1000.0
                    isi_sec = self.isi / 1000.0

                    # Record start time for precise timing
                    flash_start = time.time()

                    # Start flash
                    self.current_flash = flash_item
                    self.is_flashing = True
                    self.flash_start_time = flash_start

                    # Mark the event in the EEG stream with precise timestamp
                    # This is critically important to synchronize EEG data with visual stimuli
                    event_data = {"type": "p300_flash", "stimulus_id": flash_item}

                    if self.eeg_device and self.eeg_device.is_connected:
                        # Add metadata about the flash parameters
                        event_data.update(
                            {
                                "flash_duration_ms": self.flash_duration,
                                "isi_ms": self.isi,
                                "sequence_number": seq,
                            }
                        )

                        # Determine if this flash includes the target (useful for training)
                        if self.target_letter:
                            target_coords = self._find_letter_position(
                                self.target_letter
                            )
                            if target_coords:
                                row_idx, col_idx = target_coords
                                is_target_flash = (flash_item == f"row_{row_idx}") or (
                                    flash_item == f"col_{col_idx}"
                                )
                                event_data["is_target_flash"] = is_target_flash

                        # Mark the event with all metadata
                        self.eeg_device.mark_event("stimulus", event_data)

                    # Use a more precise timing approach
                    # Calculate how long to display the flash
                    elapsed_time = 0
                    while elapsed_time < flash_duration_sec and self.running:
                        elapsed_time = time.time() - flash_start
                        # Short sleep to prevent CPU hogging
                        time.sleep(0.001)

                    # End flash precisely when the duration has elapsed
                    self.is_flashing = False

                    # Calculate how long to wait for the inter-stimulus interval
                    # This accounts for any processing delays during the flash
                    total_elapsed = time.time() - flash_start
                    remaining_wait = max(
                        0, isi_sec - (total_elapsed - flash_duration_sec)
                    )

                    # Wait for the remaining inter-stimulus interval
                    if remaining_wait > 0 and self.running:
                        time.sleep(remaining_wait)
            # After sequences, predict character based on collected epochs
            if self.running:
                if self.state == P300SpellerState.TRAINING:
                    # In training mode, check if we need to move to next target
                    self.current_training_trial += 1
                    if self.current_training_trial >= self.required_training_trials:
                        self.finish_training()
                    else:
                        self._set_next_training_target()

                elif self.state in [
                    P300SpellerState.CLASSIFYING,
                    P300SpellerState.ADAPTING,
                ]:
                    # Use classifier to predict character
                    predicted_char, confidence = self.predict_character()

                    # Check if we've reached confidence threshold or max sequences
                    if (
                        self.dynamic_stopping_enabled
                        and confidence > self.confidence_threshold
                        and self.current_sequence_count >= self.min_sequences
                    ):

                        # Add predicted character to result
                        self.text_result += predicted_char

                        # Store for error correction
                        if self.error_correction_enabled:
                            self.undo_stack.append(predicted_char)

                        # Mark the selection event in the EEG stream
                        if self.eeg_device and self.eeg_device.is_connected:
                            self.eeg_device.mark_event(
                                "selection",
                                {
                                    "type": "character_selected",
                                    "character": predicted_char,
                                    "confidence": confidence,
                                    "sequences_used": self.current_sequence_count,
                                },
                            )

                        logger.info(
                            f"Selected character: {predicted_char} "
                            f"(confidence: {confidence:.3f}, "
                            f"sequences: {self.current_sequence_count})"
                        )

                        # Reset sequence count for next character
                        self.current_sequence_count = 0

                        # Pause before next character
                        time.sleep(2.0)

                    else:
                        # Not enough confidence yet, continue with more sequences
                        self.current_sequence_count += 1

                        # Check if we've reached maximum sequences
                        if self.current_sequence_count >= self.max_sequences:
                            # Select character with highest confidence so far
                            predicted_char, confidence = self.predict_character()
                            self.text_result += predicted_char

                            logger.info(
                                f"Selected character after max sequences: "
                                f"{predicted_char} (confidence: {confidence:.3f})"
                            )

                            # Reset for next character
                            self.current_sequence_count = 0
                            time.sleep(2.0)

    def _find_letter_position(self, letter: str) -> Tuple[int, int]:
        """Find the position (row, column) of a letter in the matrix."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] == letter:
                    return (i, j)
        return None

    def recalculate_dimensions(self) -> None:
        """Recalculate cell dimensions based on screen size to make the interface responsive."""
        # Calculate optimal cell size based on screen dimensions and matrix size
        max_cell_width = int(self.width * 0.8) // self.cols
        max_cell_height = int(self.height * 0.7) // self.rows

        # Choose the smaller dimension to maintain square cells
        cell_size = min(max_cell_width, max_cell_height)

        # Update cell dimensions
        self.cell_width = cell_size
        self.cell_height = cell_size

        # Calculate matrix position (centered)
        self.matrix_width = self.cell_width * self.cols
        self.matrix_height = self.cell_height * self.rows
        self.matrix_x = (self.width - self.matrix_width) // 2
        self.matrix_y = (self.height - self.matrix_height) // 3  # Upper third of screen

        logger.debug(
            f"Recalculated dimensions: cell={cell_size}px, matrix={self.matrix_width}x{self.matrix_height}"
        )

    def set_flash_duration(self, duration_ms: int) -> None:
        """Set the flash duration in milliseconds."""
        self.flash_duration = max(50, min(500, duration_ms))  # Limit to 50-500ms range
        logger.info(f"Flash duration set to {self.flash_duration}ms")

    def set_isi(self, isi_ms: int) -> None:
        """Set the inter-stimulus interval in milliseconds."""
        self.isi = max(50, min(500, isi_ms))  # Limit to 50-500ms range
        logger.info(f"Inter-stimulus interval set to {self.isi}ms")

    def set_color_scheme(self, scheme_name: str) -> bool:
        """Set the color scheme by name."""
        if scheme_name in COLOR_SCHEMES:
            self.color_scheme = COLOR_SCHEMES[scheme_name]
            logger.info(f"Color scheme set to {scheme_name}")
            return True
        else:
            logger.warning(f"Unknown color scheme: {scheme_name}")
            return False

    def set_flash_brightness(self, brightness: float) -> None:
        """Set the flash brightness (0.0-1.0)."""
        self.flash_brightness = min(max(brightness, 0.0), 1.0)
        logger.info(f"Flash brightness set to {self.flash_brightness}")

    def set_matrix_type(self, matrix_type: str) -> bool:
        """Set the matrix type from predefined options."""
        if matrix_type in self.ALTERNATIVE_MATRICES:
            self.matrix = self.ALTERNATIVE_MATRICES[matrix_type]
            self.matrix_type = matrix_type

            # Update dimensions based on new matrix
            self.rows = len(self.matrix)
            self.cols = len(self.matrix[0]) if self.rows > 0 else 0
            self.recalculate_dimensions()

            logger.info(f"Matrix type set to {matrix_type}")
            return True
        else:
            logger.warning(f"Unknown matrix type: {matrix_type}")
            return False

    def set_target_letter(self, letter: str) -> bool:
        """Set the current target letter for training/calibration."""
        # Find if letter exists in matrix
        pos = self._find_letter_position(letter)
        if pos:
            self.target_letter = letter
            logger.info(f"Target letter set to {letter}")
            return True
        else:
            logger.warning(f"Letter '{letter}' not found in current matrix")
            return False

    def resize(self, width: int, height: int) -> None:
        """Resize the interface to fit a different window size."""
        self.width = width
        self.height = height

        # Update the screen
        if self.screen:
            self.screen = pygame.display.set_mode((self.width, self.height))

        # Recalculate dimensions for responsive layout
        self.recalculate_dimensions()
        logger.info(f"Interface resized to {width}x{height}")

    def start_training(self) -> None:
        """Start the training phase for P300 classification."""
        if not self.classifier:
            logger.error("Cannot start training: No classifier initialized")
            return

        self.state = P300SpellerState.TRAINING
        self.training_epochs = []
        self.training_labels = []
        self.current_training_trial = 0

        # Set first training target
        self._set_next_training_target()

        logger.info("Started P300 classifier training phase")

    def _set_next_training_target(self) -> None:
        """Set the next target letter for training."""
        # Choose a random letter from the matrix
        row = random.randint(0, self.rows - 1)
        col = random.randint(0, self.cols - 1)
        target = self.matrix[row][col]
        self.set_target_letter(target)

        logger.info(f"Next training target: {target}")

    def collect_training_epoch(self, epoch: np.ndarray, is_target: bool) -> None:
        """
        Collect an epoch for training data.

        Args:
            epoch: EEG epoch data
            is_target: Whether this epoch was from a target stimulus
        """
        self.training_epochs.append(epoch)
        self.training_labels.append(1 if is_target else 0)

        logger.debug(f"Collected training epoch (target={is_target})")

    def finish_training(self) -> None:
        """
        Finish the training phase and train the classifier.
        """
        if len(self.training_epochs) < 20:
            logger.warning("Not enough training data collected")
            return

        logger.info("Training classifier...")

        # Train the classifier
        metrics = self.classifier.train(
            self.training_epochs, self.training_labels, cross_validate=True
        )

        self.state = P300SpellerState.CLASSIFYING
        logger.info(f"Classifier training complete. Metrics: {metrics}")

    def process_epoch(self, epoch: np.ndarray, flash_item: str) -> None:
        """
        Process an epoch of EEG data during character selection.

        Args:
            epoch: EEG epoch data
            flash_item: The type of flash ('row_X' or 'col_X')
        """
        if self.state == P300SpellerState.TRAINING:
            # During training, collect labeled epochs
            is_target = False
            if self.target_letter:
                target_pos = self._find_letter_position(self.target_letter)
                if target_pos:
                    row_idx, col_idx = target_pos
                    is_target = (flash_item == f"row_{row_idx}") or (
                        flash_item == f"col_{col_idx}"
                    )
            self.collect_training_epoch(epoch, is_target)

        elif self.state == P300SpellerState.CLASSIFYING:
            # During normal operation, collect epochs for classification
            if flash_item.startswith("row_"):
                self.current_row_epochs.append((epoch, flash_item))
            else:
                self.current_col_epochs.append((epoch, flash_item))

    def predict_character(self) -> Tuple[str, float]:
        """
        Predict the selected character based on collected epochs.

        Returns:
            Tuple of (predicted character, confidence)
        """
        if not self.classifier:
            return None, 0.0

        # Extract epochs from stored tuples
        row_epochs = [epoch for epoch, _ in self.current_row_epochs]
        col_epochs = [epoch for epoch, _ in self.current_col_epochs]

        # Get prediction
        predicted_char, confidence = self.classifier.predict_character(
            row_epochs, col_epochs, self.matrix
        )

        # Clear current epochs
        self.current_row_epochs = []
        self.current_col_epochs = []

        # Update classifier with online adaptation if confidence is high
        if (
            confidence > self.confidence_threshold
            and self.state == P300SpellerState.ADAPTING
        ):
            self._update_classifier_online(predicted_char)

        return predicted_char, confidence

    def _update_classifier_online(self, predicted_char: str) -> None:
        """
        Update classifier with online adaptation.

        Args:
            predicted_char: The character that was predicted
        """
        if not self.classifier:
            return

        # Collect all epochs from the last prediction
        all_epochs = []
        all_labels = []

        # Process row epochs
        for epoch, flash_item in self.current_row_epochs:
            row_idx = int(flash_item.split("_")[1])
            is_target = (
                predicted_char == self.matrix[row_idx][0]
            )  # Any column would work
            all_epochs.append(epoch)
            all_labels.append(1 if is_target else 0)

        # Process column epochs
        for epoch, flash_item in self.current_col_epochs:
            col_idx = int(flash_item.split("_")[1])
            is_target = any(
                predicted_char == self.matrix[i][col_idx] for i in range(self.rows)
            )
            all_epochs.append(epoch)
            all_labels.append(1 if is_target else 0)

        # Update classifier
        for epoch, label in zip(all_epochs, all_labels):
            self.classifier.adapt_online(epoch, label)

        logger.debug("Updated classifier with online adaptation")

    def update_performance_metrics(self, correct):
        """Update performance metrics and adjust timing."""
        self.performance_window.append(correct)
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)

        # Update overall statistics
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1

        # Adjust ISI based on recent performance
        if len(self.performance_window) >= self.window_size:
            recent_accuracy = sum(self.performance_window) / len(
                self.performance_window
            )
            self.adapt_timing(recent_accuracy)

    def adapt_timing(self, accuracy):
        """Adapt stimulus timing based on performance."""
        if accuracy > 0.9:  # Very good performance
            self.isi = max(75, self.isi - 10)  # Speed up but not too fast
        elif accuracy < 0.7:  # Poor performance
            self.isi = min(200, self.isi + 10)  # Slow down but not too slow
        # Between 0.7 and 0.9 maintain current speed

    def draw_performance_feedback(self):
        """Draw performance metrics on screen."""
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            speed_text = f"Speed: {self.isi}ms"

            font = pygame.font.Font(None, 24)
            acc_surface = font.render(f"Accuracy: {accuracy:.1f}%", True, WHITE)
            speed_surface = font.render(speed_text, True, WHITE)

            self.screen.blit(acc_surface, (10, 10))
            self.screen.blit(speed_surface, (10, 40))

    def flash_next_stimulus(self):
        """Flash the next stimulus in sequence with visual enhancement."""
        # Reset previously flashed items
        if self.current_flash:
            self.reset_flash()

        # Get next flash coordinates
        self.current_flash = self.get_next_flash()
        if self.current_flash:
            self.enhance_flash()

    def reset_flash(self):
        """Reset a flashed item back to its normal state."""
        self.current_flash = None
        self.is_flashing = False

    def enhance_flash(self):
        """Enhanced visual feedback for flashed items."""
        if not self.current_flash:
            return

        flash_type, idx = self.current_flash.split("_")
        idx = int(idx)

        # Create a gradient effect
        alpha = 255
        flash_color = (255, 255, 255, alpha)

        # Get the cells to flash based on whether it's a row or column
        cells_to_flash = []
        if flash_type == "row":
            for col in range(self.cols):
                cells_to_flash.append((idx, col))
        else:  # column
            for row in range(self.rows):
                cells_to_flash.append((row, idx))

        # Flash each cell
        for row, col in cells_to_flash:
            rect = self.get_cell_rect(row, col)
            flash_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(flash_surface, flash_color, flash_surface.get_rect())
            self.screen.blit(flash_surface, rect)

    def get_cell_rect(self, row, col):
        """Get the rectangle area of a cell in the matrix."""
        return pygame.Rect(
            self.matrix_x + col * self.cell_width,
            self.matrix_y + row * self.cell_height,
            self.cell_width,
            self.cell_height,
        )

    def initialize_performance_tracking(self) -> None:
        """Initialize performance tracking metrics."""
        # Initialize basic metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.window_size = self.performance_window_size
        self.last_flash_time = pygame.time.get_ticks()

        # Initialize adaptive timing parameters
        self.current_accuracy = 0.0

        # Reset performance windows
        self.performance_window = []  # For storing recent accuracy values
        self.accuracy_history = []  # For long-term tracking

        logger.debug("Performance tracking initialized")

    def eeg_data_available(self) -> bool:
        """Check if new EEG data is available for processing."""
        if (
            not hasattr(self, "eeg_device")
            or not self.eeg_device
            or not self.eeg_device.is_connected
        ):
            return False

        return True  # Data is always being streamed when device is connected

    def process_eeg_data(self) -> Optional[str]:
        """Process available EEG data and return prediction if available."""
        if (
            not hasattr(self, "eeg_device")
            or not self.eeg_device
            or not hasattr(self, "classifier")
            or not self.classifier
        ):
            logger.error("Required components not initialized")
            return None

        if not hasattr(self.classifier, "is_fitted") or not self.classifier.is_fitted:
            logger.warning("Classifier not fitted yet - run training first")
            return None

        try:
            # Update the buffer first
            if not self.eeg_device.update_buffer():
                logger.debug("Buffer update failed")
                return None

            # Get the latest data from the buffer
            data = self.eeg_device.data_buffer

            if data is None or data.size == 0:
                logger.debug("Empty data buffer")
                return None

            # Ensure we have enough samples
            if data.shape[1] < self.eeg_device.buffer_size:
                logger.debug(
                    f"Not enough samples yet: {data.shape[1]}/{self.eeg_device.buffer_size}"
                )
                return None

            # Validate data before preprocessing
            if not np.all(np.isfinite(data)):
                logger.warning("Invalid values in raw data, cleaning...")
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Preprocess the data if we have a preprocessor
            if hasattr(self, "preprocessor") and self.preprocessor:
                try:
                    data, preprocess_metadata = self.preprocessor.process_chunk(
                        data, is_stimulus=True
                    )
                    if preprocess_metadata.get("artifacts_detected", False):
                        logger.debug("Artifacts detected and removed")
                except Exception as e:
                    logger.error(f"Preprocessing error: {e}")
                    return None

            # Validate processed data before classification
            if data is None or data.size == 0:
                logger.error("Preprocessing resulted in empty data")
                return None

            # Process the data through the classifier
            try:
                prediction, confidence = self.classifier.predict(data)

                if confidence > self.min_confidence_threshold:
                    logger.debug(f"Prediction made with confidence {confidence:.3f}")
                    return prediction
                else:
                    logger.debug(f"Confidence too low: {confidence:.3f}")

            except Exception as e:
                logger.error(f"Classification error: {e}")

        except Exception as e:
            logger.error(f"Error processing EEG data: {e}")
            import traceback

            logger.debug(traceback.format_exc())

        return None

    def get_next_flash(self) -> str:
        """
        Get the next item to flash in sequence.

        Returns:
            String identifier of the next flash (e.g. 'row_0' or 'col_1')
        """
        # In a real sequence, this would come from the stimulus presentation thread
        # For now, randomly select a row or column
        if random.random() < 0.5:
            # Flash a row
            row = random.randint(0, self.rows - 1)
            return f"row_{row}"
        else:
            # Flash a column
            col = random.randint(0, self.cols - 1)
            return f"col_{col}"


# Example usage
if __name__ == "__main__":
    try:
        # Create the EEG device connection
        eeg_device = UnicornP300Speller()

        # Create and start the P300 speller interface
        speller = P300SpellerInterface(eeg_device=eeg_device)
        speller.start()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean up resources
        if "speller" in locals():
            speller.stop()
