#!/usr/bin/env python3
# Advanced Guitar and Bass Analyzer with Tab Generation
# For professional-grade music analysis

import os
import sys
import time
import json
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
from pathlib import Path

class GuitarAnalyzer:
    """
    Advanced analyzer for guitar and bass tracks that can detect
    notes, chords, and generate tablature.
    
    Features:
    - Pitch detection optimized for guitar/bass frequencies
    - Multi-pitch detection for chords
    - String and fret position estimation
    - Tab generation with proper timing
    - Playing technique detection (bends, slides, etc.)
    - Support for alternate tunings
    """
    
    # Standard tunings (from lowest to highest string)
    STANDARD_TUNINGS = {
        'guitar': ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
        'bass': ['E1', 'A1', 'D2', 'G2'],
        'guitar_7string': ['B1', 'E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
        'bass_5string': ['B0', 'E1', 'A1', 'D2', 'G2'],
        'guitar_drop_d': ['D2', 'A2', 'D3', 'G3', 'B3', 'E4'],
        'bass_drop_d': ['D1', 'A1', 'D2', 'G2']
    }
    
    # Frequency ranges for each instrument
    FREQUENCY_RANGES = {
        'guitar': (80, 1200),  # Hz (Low E2 to high notes with harmonics)
        'bass': (30, 400)      # Hz (Low B0 to high notes on G string)
    }
    
    def __init__(self, instrument='guitar', tuning=None, frets=24):
        """
        Initialize the guitar/bass analyzer.
        
        Parameters:
        - instrument: 'guitar' or 'bass'
        - tuning: List of notes from lowest to highest string, or a key from STANDARD_TUNINGS
        - frets: Number of frets on the instrument
        """
        # Basic setup
        self.instrument = instrument.lower()
        self.frets = frets
        
        # Set appropriate frequency range based on instrument
        if instrument.lower() not in self.FREQUENCY_RANGES:
            print(f"Warning: Unknown instrument '{instrument}', defaulting to guitar")
            self.instrument = 'guitar'
            
        self.freq_range = self.FREQUENCY_RANGES[self.instrument]
        
        # Set tuning
        if tuning is None:
            # Default to standard tuning
            if self.instrument == 'guitar':
                self.tuning = self.STANDARD_TUNINGS['guitar']
            elif self.instrument == 'bass':
                self.tuning = self.STANDARD_TUNINGS['bass']
            else:
                self.tuning = self.STANDARD_TUNINGS['guitar']
        elif isinstance(tuning, str) and tuning in self.STANDARD_TUNINGS:
            # Use a predefined tuning
            self.tuning = self.STANDARD_TUNINGS[tuning]
        else:
            # Use custom tuning
            self.tuning = tuning
            
        # Calculate string frequencies based on tuning
        self.string_frequencies = self._calculate_string_frequencies()
        
        # Initialize state variables
        self.audio = None
        self.sr = None
        self.note_events = []
        self.chord_events = []
        self.tablature = []
        
        # Technique detection flags
        self.detect_bends = True
        self.detect_slides = True
        self.detect_hammer_pulls = True
        self.detect_palm_muting = True
        
        print(f"Initialized {self.instrument} analyzer with {len(self.tuning)} strings in tuning: {', '.join(self.tuning)}")
    
    def _calculate_string_frequencies(self):
        """Calculate the fundamental frequencies for each open string."""
        string_frequencies = []
        
        for note in self.tuning:
            # Extract the note name and octave
            if note[-1].isdigit():  # If note has octave number (e.g., E2)
                note_name = note[:-1]
                octave = int(note[-1])
            else:  # Default to octave 4 if not specified
                note_name = note
                octave = 4
            
            # Convert note to frequency (using librosa's note_to_hz)
            freq = librosa.note_to_hz(f"{note_name}{octave}")
            string_frequencies.append(freq)
            
        return string_frequencies
    
    def _calculate_fret_frequencies(self):
        """Calculate frequencies for all frets on all strings."""
        fret_frequencies = []
        
        for string_idx, open_freq in enumerate(self.string_frequencies):
            string_frets = [open_freq]  # Start with open string
            
            # Calculate frequency for each fret
            for fret in range(1, self.frets + 1):
                # Each fret is a semitone, which is a factor of 2^(1/12)
                freq = open_freq * (2 ** (fret / 12))
                string_frets.append(freq)
                
            fret_frequencies.append(string_frets)
            
        return fret_frequencies
    
    def load_audio(self, file_path):
        """Load an audio file containing guitar or bass for analysis."""
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            self.audio = y
            self.sr = sr
            self.duration = librosa.get_duration(y=y, sr=sr)
            
            print(f"Loaded {self.instrument} audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            
            # Apply a bandpass filter to focus on instrument frequency range
            self.filtered_audio = self._bandpass_filter(self.audio, self.freq_range[0], self.freq_range[1])
            
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def _bandpass_filter(self, audio, low_freq, high_freq):
        """Apply a bandpass filter to focus on relevant frequencies."""
        # Use librosa's resampling to create a simple bandpass effect
        lowpass = librosa.resample(
            audio, 
            orig_sr=self.sr, 
            target_sr=high_freq * 2  # Nyquist frequency
        )
        bandpass = librosa.resample(
            lowpass, 
            orig_sr=high_freq * 2,
            target_sr=self.sr
        )
        
        # Now filter out very low frequencies
        if low_freq > 0:
            highpass = librosa.effects.preemphasis(bandpass, coef=0.95 * low_freq / (self.sr / 2))
            return highpass
        
        return bandpass
    
    def detect_notes(self, frame_length=4096, hop_length=1024, threshold=0.5):
        """
        Detect individual notes played on the instrument.
        
        Parameters:
        - frame_length: Size of the analysis window
        - hop_length: Number of samples between successive frames
        - threshold: Threshold for peak detection
        
        Returns a list of note events, each with timing, frequency, and estimated string/fret.
        """
        if self.audio is None:
            print("No audio loaded. Please load an audio file first.")
            return []
        
        # Storage for detected notes
        self.note_events = []
        
        # Calculate fret frequencies for all possible positions
        fret_frequencies = self._calculate_fret_frequencies()
        
        # Compute the harmonic pitch class profile (HPCP) for the whole audio
        # This gives us the distribution of pitch classes over time
        chromagram = librosa.feature.chroma_cqt(
            y=self.filtered_audio, 
            sr=self.sr,
            hop_length=hop_length,
            n_chroma=12,
            bins_per_octave=36  # Higher resolution for more accuracy
        )
        
        # Compute spectrogram with higher resolution for precise pitch estimation
        spec = np.abs(librosa.stft(self.filtered_audio, n_fft=frame_length, hop_length=hop_length))
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=frame_length)
        
        # Get timing information
        times = librosa.times_like(chromagram, sr=self.sr, hop_length=hop_length)
        
        # Detect onsets for note segmentation
        onset_env = librosa.onset.onset_strength(
            y=self.filtered_audio, sr=self.sr, hop_length=hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=hop_length,
            units='frames'
        )
        
        # Convert onsets to time
        onset_times = librosa.frames_to_time(onsets, sr=self.sr, hop_length=hop_length)
        
        # Process each onset
        for i in range(len(onsets)):
            # Get the start frame of this note
            start_frame = onsets[i]
            
            # Determine the end frame (either the next onset or a fixed duration)
            if i < len(onsets) - 1:
                end_frame = onsets[i + 1]
            else:
                # For the last note, use a fixed duration or end of file
                end_frame = min(start_frame + 8, spec.shape[1] - 1)  # 8 frames ≈ 200ms
            
            # Get the spectrum for this note
            note_spec = spec[:, start_frame:end_frame]
            
            # Find the most dominant frequencies in this note segment
            if note_spec.size > 0:
                frame_energies = np.sum(note_spec, axis=1)
                # Only consider frequencies in our instrument's range
                valid_range = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
                frame_energies = frame_energies * valid_range
                
                # Find peaks in the spectrum (dominant frequencies)
                peaks, _ = find_peaks(frame_energies, height=threshold * np.max(frame_energies))
                peak_freqs = freqs[peaks]
                peak_mags = frame_energies[peaks]
                
                # Sort by magnitude
                if len(peak_freqs) > 0:
                    sorted_idx = np.argsort(peak_mags)[::-1]  # Descending order
                    peak_freqs = peak_freqs[sorted_idx]
                    peak_mags = peak_mags[sorted_idx]
                    
                    # For each peak, estimate the most likely string and fret
                    for freq, mag in zip(peak_freqs[:3], peak_mags[:3]):  # Only use top 3 peaks
                        string, fret = self._estimate_string_and_fret(freq)
                        
                        # Calculate the confidence based on how close the frequency is to the estimated fret
                        estimated_freq = self.string_frequencies[string] * (2 ** (fret / 12))
                        confidence = 1.0 - abs(freq - estimated_freq) / estimated_freq
                        
                        # Only add if confidence is reasonable
                        if confidence > 0.85:
                            note_event = {
                                'time': times[start_frame],
                                'duration': times[min(end_frame, len(times)-1)] - times[start_frame],
                                'frequency': float(freq),
                                'magnitude': float(mag),
                                'string': int(string),
                                'fret': int(fret),
                                'confidence': float(confidence)
                            }
                            self.note_events.append(note_event)
        
        # Detect techniques (bends, slides, etc.)
        if self.detect_bends or self.detect_slides:
            self._detect_playing_techniques()
        
        print(f"Detected {len(self.note_events)} note events")
        return self.note_events
    
    def _estimate_string_and_fret(self, frequency):
        """
        Estimate the most likely string and fret for a given frequency.
        
        Returns a tuple of (string_index, fret_number).
        """
        best_string = 0
        best_fret = 0
        min_distance = float('inf')
        
        # Try each string
        for string_idx, open_freq in enumerate(self.string_frequencies):
            # Calculate which fret this would be on the current string
            if frequency < open_freq:
                # Frequency is below the open string, can't be played on this string
                continue
                
            # Calculate the fret that would produce this frequency
            # Using the formula: fret = 12 * log2(freq / open_freq)
            fret = 12 * np.log2(frequency / open_freq)
            
            # Find the closest actual fret (since fret might not be a whole number)
            closest_fret = round(fret)
            
            # Check if the fret is within the range of the instrument
            if 0 <= closest_fret <= self.frets:
                # Calculate the actual frequency of this fret
                actual_freq = open_freq * (2 ** (closest_fret / 12))
                
                # Calculate how far this is from the target frequency
                distance = abs(frequency - actual_freq)
                
                # If this is the closest match so far, store it
                if distance < min_distance:
                    min_distance = distance
                    best_string = string_idx
                    best_fret = closest_fret
        
        return best_string, best_fret
    
    def _detect_playing_techniques(self):
        """
        Detect various playing techniques like bends, slides, hammer-ons, etc.
        Modifies note_events in place to add technique information.
        """
        if len(self.note_events) < 2:
            return
        
        # Sort note events by time
        sorted_events = sorted(self.note_events, key=lambda x: x['time'])
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Skip if events are on different strings
            if current['string'] != next_event['string']:
                continue
            
            # Detect bends
            if self.detect_bends:
                # A bend typically shows as a gradual increase in frequency on the same string
                # within a short time window
                time_diff = next_event['time'] - current['time']
                if time_diff < 0.3:  # Less than 300ms apart
                    freq_ratio = next_event['frequency'] / current['frequency']
                    
                    # Check if frequency increased by approximately a semitone or more
                    if 1.05 < freq_ratio < 1.2:  # Quarter bend to full semitone
                        current['technique'] = 'quarter_bend'
                    elif 1.2 <= freq_ratio < 1.5:  # Full semitone to whole tone
                        current['technique'] = 'half_bend'
                    elif freq_ratio >= 1.5:  # More than a whole tone
                        current['technique'] = 'full_bend'
            
            # Detect slides
            if self.detect_slides:
                # A slide typically involves the same string but different frets in sequence
                time_diff = next_event['time'] - current['time']
                if 0.05 < time_diff < 0.5:  # Appropriate timing for a slide
                    fret_diff = next_event['fret'] - current['fret']
                    
                    # Slides involve moving up or down the fretboard
                    if abs(fret_diff) > 1:
                        if fret_diff > 0:
                            current['technique'] = 'slide_up'
                        else:
                            current['technique'] = 'slide_down'
            
            # Detect hammer-ons and pull-offs
            if self.detect_hammer_pulls:
                time_diff = next_event['time'] - current['time']
                if time_diff < 0.15:  # Very quick succession
                    fret_diff = next_event['fret'] - current['fret']
                    
                    if fret_diff > 0 and fret_diff <= 5:
                        current['technique'] = 'hammer_on'
                    elif fret_diff < 0 and fret_diff >= -5:
                        current['technique'] = 'pull_off'
    
    def detect_chords(self, window_size=3, min_notes=2, min_confidence=0.7):
        """
        Detect chords from the note events.
        
        Parameters:
        - window_size: Time window in which to group notes into a chord (in frames)
        - min_notes: Minimum number of notes to be considered a chord
        - min_confidence: Minimum average confidence for a valid chord
        
        Returns a list of chord events with timing and component notes.
        """
        if not self.note_events:
            print("No note events detected. Please run detect_notes() first.")
            return []
        
        # Sort notes by time
        sorted_notes = sorted(self.note_events, key=lambda x: x['time'])
        
        # Group notes into potential chords
        current_chord = []
        chord_groups = []
        
        for note in sorted_notes:
            if not current_chord:
                # First note in potential chord
                current_chord = [note]
            else:
                # Check if this note is within the time window of the current chord
                time_diff = note['time'] - current_chord[0]['time']
                
                if time_diff <= window_size / self.sr:
                    # Add to current chord
                    current_chord.append(note)
                else:
                    # Outside time window, finish current chord and start a new one
                    if len(current_chord) >= min_notes:
                        chord_groups.append(current_chord)
                    current_chord = [note]
        
        # Add the last chord if it has enough notes
        if len(current_chord) >= min_notes:
            chord_groups.append(current_chord)
        
        # Process each chord group to create chord events
        for group in chord_groups:
            # Calculate average confidence
            avg_confidence = sum(note['confidence'] for note in group) / len(group)
            
            if avg_confidence >= min_confidence:
                # Extract unique string/fret combinations (avoid duplicates from harmonics)
                string_frets = set((note['string'], note['fret']) for note in group)
                
                # Create the chord event
                chord_event = {
                    'time': min(note['time'] for note in group),
                    'duration': max(note['time'] + note['duration'] for note in group) - min(note['time'] for note in group),
                    'notes': [{'string': sf[0], 'fret': sf[1]} for sf in string_frets],
                    'confidence': avg_confidence
                }
                
                # Try to identify the chord
                chord_name = self._identify_chord(chord_event['notes'])
                if chord_name:
                    chord_event['name'] = chord_name
                
                self.chord_events.append(chord_event)
        
        print(f"Detected {len(self.chord_events)} chord events")
        return self.chord_events
    
    def _identify_chord(self, notes):
        """
        Attempt to identify the chord name from the set of notes.
        
        Parameters:
        - notes: List of {'string': string_idx, 'fret': fret_num} dictionaries
        
        Returns the chord name or None if unidentified.
        """
        if len(notes) < 2:
            return None
        
        # Get the pitch classes (0-11) for each note
        pitch_classes = set()
        for note in notes:
            string_idx = note['string']
            fret = note['fret']
            
            # Calculate the frequency
            freq = self.string_frequencies[string_idx] * (2 ** (fret / 12))
            
            # Convert to MIDI note number and then to pitch class
            midi_note = librosa.hz_to_midi(freq)
            pitch_class = int(midi_note) % 12
            pitch_classes.add(pitch_class)
        
        # Common chord structures (as pitch class intervals from root)
        chord_types = {
            # Major chords
            frozenset([0, 4, 7]): 'maj',           # Major triad
            frozenset([0, 4, 7, 11]): 'maj7',      # Major seventh
            frozenset([0, 4, 7, 10]): '7',         # Dominant seventh
            
            # Minor chords
            frozenset([0, 3, 7]): 'min',           # Minor triad
            frozenset([0, 3, 7, 10]): 'min7',      # Minor seventh
            frozenset([0, 3, 7, 11]): 'min/maj7',  # Minor-major seventh
            
            # Augmented and diminished
            frozenset([0, 4, 8]): 'aug',           # Augmented
            frozenset([0, 3, 6]): 'dim',           # Diminished
            frozenset([0, 3, 6, 9]): 'dim7',       # Diminished seventh
            
            # Suspended chords
            frozenset([0, 5, 7]): 'sus4',          # Suspended 4th
            frozenset([0, 2, 7]): 'sus2',          # Suspended 2nd
        }
        
        # Normalize the pitch classes to find the root
        best_match = None
        best_score = 0
        
        for root in range(12):
            # Shift all pitch classes to normalize to this root
            normalized = frozenset((pc - root) % 12 for pc in pitch_classes)
            
            # Try to match with known chord types
            for template, name in chord_types.items():
                # Calculate match score (number of matching notes divided by total distinct notes)
                matches = len(normalized.intersection(template))
                total = len(normalized.union(template))
                score = matches / total
                
                if score > best_score:
                    best_score = score
                    note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root]
                    best_match = f"{note_name}{name}"
        
        # Only return a match if the score is high enough
        if best_score >= 0.7:
            return best_match
        
        return None
    
    def generate_tablature(self, use_chords=True, optimize_positions=True, include_techniques=True):
        """
        Generate guitar/bass tablature from the detected notes and chords.
        
        Parameters:
        - use_chords: Whether to use detected chords or individual notes
        - optimize_positions: Try to optimize fret positions for playability
        - include_techniques: Include detected techniques in the tab
        
        Returns a list of tab lines, each representing a moment in time.
        """
        if not self.note_events and not self.chord_events:
            print("No notes or chords detected. Please run detect_notes() or detect_chords() first.")
            return []
        
        # Determine the events to use
        if use_chords and self.chord_events:
            events = self.chord_events
            event_type = 'chord'
        else:
            events = self.note_events
            event_type = 'note'
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x['time'])
        
        # Create empty tab representation for each string
        num_strings = len(self.tuning)
        tab_duration = max(event['time'] + event['duration'] for event in events) + 1.0
        
        # Determine the number of tab columns (based on time resolution)
        time_resolution = 0.25  # Quarter note resolution
        num_columns = int(tab_duration / time_resolution) + 1
        
        # Initialize tab with empty strings
        tab = [[''] * num_columns for _ in range(num_strings)]
        
        # Map events to tab positions
        for event in sorted_events:
            # Calculate the column index for this event
            col_idx = int(event['time'] / time_resolution)
            
            if col_idx < num_columns:
                if event_type == 'chord':
                    # For chords, place each note on its string
                    for note in event['notes']:
                        string_idx = note['string']
                        fret = note['fret']
                        
                        if string_idx < num_strings:
                            # Basic fret notation
                            notation = str(fret)
                            
                            # Add technique notation if available
                            if include_techniques and 'technique' in event:
                                technique = event['technique']
                                if technique == 'hammer_on':
                                    notation = f"{notation}h"
                                elif technique == 'pull_off':
                                    notation = f"{notation}p"
                                elif technique == 'slide_up':
                                    notation = f"{notation}/"
                                elif technique == 'slide_down':
                                    notation = f"{notation}\\"
                                elif technique.endswith('_bend'):
                                    notation = f"{notation}b"
                            
                            tab[string_idx][col_idx] = notation
                            
                else:  # note event
                    string_idx = event['string']
                    fret = event['fret']
                    
                    if string_idx < num_strings:
                        # Basic fret notation
                        notation = str(fret)
                        
                        # Add technique notation if available
                        if include_techniques and 'technique' in event:
                            technique = event['technique']
                            if technique == 'hammer_on':
                                notation = f"{notation}h"
                            elif technique == 'pull_off':
                                notation = f"{notation}p"
                            elif technique == 'slide_up':
                                notation = f"{notation}/"
                            elif technique == 'slide_down':
                                notation = f"{notation}\\"
                            elif technique.endswith('_bend'):
                                notation = f"{notation}b"
                        
                        tab[string_idx][col_idx] = notation
        
        # Build the text representation of the tab
        tab_lines = []
        
        # Add tuning information
        tab_lines.append(f"# {self.instrument.capitalize()} Tab - Tuning: {', '.join(self.tuning)}")
        tab_lines.append("")
        
        # Add tab notation
        for string_idx, string in enumerate(tab):
            # Add the string line
            line = f"{self.tuning[string_idx]}|-"
            
            for col in string:
                if col:
                    # Add the fret number, padding to 2 characters
                    if len(col) == 1:
                        line += f"{col}-"
                    else:
                        line += f"{col}"
                else:
                    # Empty position
                    line += "--"
            
            line += "|"
            tab_lines.append(line)
        
        self.tablature = tab_lines
        return tab_lines
    
    def export_tablature(self, filename="tab.txt"):
        """Export the tablature to a text file."""
        if not self.tablature:
            print("No tablature generated. Please run generate_tablature() first.")
            return False
        
        try:
            with open(filename, 'w') as f:
                for line in self.tablature:
                    f.write(f"{line}\n")
            
            print(f"Tablature saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving tablature: {e}")
            return False
    
    def visualize_notes(self, output_file="notes_visualization.png"):
        """Visualize the detected notes on the fretboard over time."""
        if not self.note_events:
            print("No notes detected. Please run detect_notes() first.")
            return False
        
        try:
            plt.figure(figsize=(15, 8))
            
            # Plot the audio waveform at the top
            plt.subplot(211)
            librosa.display.waveshow(self.filtered_audio, sr=self.sr)
            plt.title(f"{self.instrument.capitalize()} Audio Waveform")
            
            # Plot note events as colored points on a grid
            plt.subplot(212)
            
            # Create a scatter plot of detected notes
            # x-axis: time, y-axis: string and fret
            times = [note['time'] for note in self.note_events]
            
            # Calculate y positions based on string and fret
            # Higher string numbers should appear at the top (lower y value)
            num_strings = len(self.tuning)
            y_positions = []
            colors = []
            sizes = []
            labels = []
            
            for note in self.note_events:
                # Calculate position: string * 10 + fret
                string_idx = note['string']
                fret = note['fret']
                
                # Invert the string index so lower strings are at the bottom
                inverted_string = num_strings - 1 - string_idx
                position = inverted_string * (self.frets + 1) + fret
                
                y_positions.append(position)
                
                # Color based on confidence
                colors.append(note['confidence'])
                
                # Size based on magnitude
                sizes.append(50 * note['magnitude'] / max(n['magnitude'] for n in self.note_events))
                
                # Label
                if 'technique' in note:
                    labels.append(f"String {string_idx+1}, Fret {fret}, {note['technique']}")
                else:
                    labels.append(f"String {string_idx+1}, Fret {fret}")
            
            # Create a colormap to show the confidence levels
            scatter = plt.scatter(times, y_positions, c=colors, s=sizes, cmap='viridis', 
                                  alpha=0.8, edgecolors='k')
            
            # Add a colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Confidence')
            
            # Add string lines and labels
            for i in range(num_strings):
                inverted_i = num_strings - 1 - i
                y_pos = inverted_i * (self.frets + 1)
                plt.axhline(y=y_pos, color='gray', linestyle='-', alpha=0.5)
                plt.text(0, y_pos, self.tuning[i], fontsize=10, ha='right', va='center')
            
            # Set y-axis tick positions to show fret numbers
            string_positions = [num_strings - 1 - i for i in range(num_strings)]
            string_positions = [(pos * (self.frets + 1)) for pos in string_positions]
            plt.yticks(string_positions, [f"String {i+1}" for i in range(num_strings)])
            
            plt.xlabel('Time (seconds)')
            plt.title('Detected Notes on Fretboard')
            plt.grid(axis='y', alpha=0.3)
            
            # Limit x-axis to the duration of the audio
            plt.xlim(0, self.duration)
            
            plt.tight_layout()
            plt.savefig(output_file)
            print(f"Visualization saved as '{output_file}'")
            return True
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False
    
    def export_to_json(self, filename="guitar_analysis.json"):
        """Export detection results to JSON format."""
        if not self.note_events and not self.chord_events:
            print("No detection results available.")
            return False
        
        # Prepare data for export
        export_data = {
            'instrument': self.instrument,
            'tuning': self.tuning,
            'note_events': self.note_events,
            'chord_events': self.chord_events
        }
        
        # If tablature exists, include it
        if self.tablature:
            export_data['tablature'] = self.tablature
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Analysis results exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False


def main():
    """Command-line interface for the guitar analyzer."""
    parser = argparse.ArgumentParser(description="Guitar and Bass Analyzer with Tab Generation")
    
    parser.add_argument("audio_file", help="Path to the guitar/bass audio file")
    
    parser.add_argument("--instrument", choices=["guitar", "bass"], default="guitar",
                        help="Type of instrument in the audio")
    
    parser.add_argument("--tuning", default=None,
                        help="Instrument tuning (e.g., 'standard', 'drop_d', or comma-separated notes)")
    
    parser.add_argument("--frets", type=int, default=24,
                        help="Number of frets on the instrument")
    
    parser.add_argument("--detect-chords", action="store_true",
                        help="Detect chords in addition to individual notes")
    
    parser.add_argument("--optimize-tab", action="store_true",
                        help="Optimize tablature for playability")
    
    parser.add_argument("--no-techniques", action="store_true",
                        help="Don't detect playing techniques (bends, slides, etc.)")
    
    parser.add_argument("--output-dir", default="guitar_output",
                        help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File {args.audio_file} not found.")
        return
    
    # Parse tuning if provided
    custom_tuning = None
    if args.tuning:
        if args.tuning in GuitarAnalyzer.STANDARD_TUNINGS:
            custom_tuning = args.tuning  # Use a predefined tuning
        else:
            # Parse comma-separated notes
            custom_tuning = args.tuning.split(',')
    
    # Create analyzer
    analyzer = GuitarAnalyzer(
        instrument=args.instrument,
        tuning=custom_tuning,
        frets=args.frets
    )
    
    # Set technique detection
    if args.no_techniques:
        analyzer.detect_bends = False
        analyzer.detect_slides = False
        analyzer.detect_hammer_pulls = False
        analyzer.detect_palm_muting = False
    
    # Load audio
    if not analyzer.load_audio(args.audio_file):
        print("Failed to load audio file.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    print("\nDetecting notes...")
    analyzer.detect_notes()
    
    # Detect chords if requested
    if args.detect_chords:
        print("\nDetecting chords...")
        analyzer.detect_chords()
    
    # Visualize the notes
    print("\nGenerating visualization...")
    analyzer.visualize_notes(output_file=os.path.join(args.output_dir, "note_detection.png"))
    
    # Generate tablature
    print("\nGenerating tablature...")
    tab_lines = analyzer.generate_tablature(
        use_chords=args.detect_chords,
        optimize_positions=args.optimize_tab,
        include_techniques=not args.no_techniques
    )
    
    # Export results
    analyzer.export_tablature(filename=os.path.join(args.output_dir, "tablature.txt"))
    analyzer.export_to_json(filename=os.path.join(args.output_dir, "guitar_analysis.json"))
    
    print(f"\nComplete! All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
        