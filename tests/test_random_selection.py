#!/usr/bin/env python3
"""
Test script to verify random character selection is working properly.
"""

from speller.gui.gui_utils import default_chars
import random
import time

def test_random_selection():
    """Test random character selection to see if it's always 'A'."""
    chars = default_chars(6, 6)
    print(f"Character matrix: {chars}")
    print(f"First character (index 0): '{chars[0]}'")
    print(f"Total characters: {len(chars)}")
    print()
    
    # Test multiple random selections
    selections = []
    for i in range(10):
        # Ensure better randomness by seeding with current time
        random.seed(int(time.time() * 1000000 + i) % 2**32)
        random_char = random.choice(chars)
        char_index = chars.index(random_char)
        selections.append((random_char, char_index))
        print(f"Selection {i+1}: '{random_char}' (index {char_index})")
        time.sleep(0.001)  # Small delay to ensure different seeds
    
    # Check if all selections are the same
    unique_chars = set([s[0] for s in selections])
    print(f"\nUnique characters selected: {unique_chars}")
    print(f"Number of unique selections: {len(unique_chars)}")
    
    if len(unique_chars) == 1 and 'A' in unique_chars:
        print("❌ PROBLEM: All selections were 'A' - randomness is not working!")
    elif len(unique_chars) == 1:
        print(f"❌ PROBLEM: All selections were '{list(unique_chars)[0]}' - randomness is not working!")
    else:
        print("✅ Randomness appears to be working correctly!")
    
    return selections

if __name__ == "__main__":
    test_random_selection()
