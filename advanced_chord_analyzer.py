#!/usr/bin/env python3
# Advanced Chord Analyzer with Complex Chord Recognition
# This version includes enhanced chord detection with 7th, 9th, sus4 chords and more

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import os
import sys
import json

class AdvancedChordAnalyzer:
    """An enhanced class for analyzing and recognizing complex chords in audio files."""
    
    # Note names for reference
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, hop_length=512, n_fft=2048):
        """Initialize the analyzer with processing parameters."""
        self.hop_length = hop_length
        self.n_fft = n_fft
        self._load_chord_templates()
        
    def _load_chord_templates(self):
        """Load chord templates including complex chords."""
        # Create basic chord templates
        self.chord_templates = {}
        
        # Generate all 12 root positions for each chord type
        for root_idx, root_note in enumerate(self.NOTES):
            # Major triad (1, 3, 5)
            major_template = np.zeros(12)
            major_template[root_idx] = 1.0                        # Root
            major_template[(root_idx + 4) % 12] = 0.8             # Major third
            major_template[(root_idx + 7) % 12] = 0.8             # Perfect fifth
            self.chord_templates[f"{root_note}"] = major_template
            
            # Minor triad (1, b3, 5)
            minor_template = np.zeros(12)
            minor_template[root_idx] = 1.0                        # Root
            minor_template[(root_idx + 3) % 12] = 0.8             # Minor third
            minor_template[(root_idx + 7) % 12] = 0.8             # Perfect fifth
            self.chord_templates[f"{root_note}m"] = minor_template
            
            # Dominant 7th (1, 3, 5, b7)
            dom7_template = np.zeros(12)
            dom7_template[root_idx] = 1.0                         # Root
            dom7_template[(root_idx + 4) % 12] = 0.8              # Major third
            dom7_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            dom7_template[(root_idx + 10) % 12] = 0.7             # Minor seventh
            self.chord_templates[f"{root_note}7"] = dom7_template
            
            # Major 7th (1, 3, 5, 7)
            maj7_template = np.zeros(12)
            maj7_template[root_idx] = 1.0                         # Root
            maj7_template[(root_idx + 4) % 12] = 0.8              # Major third
            maj7_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            maj7_template[(root_idx + 11) % 12] = 0.7             # Major seventh
            self.chord_templates[f"{root_note}maj7"] = maj7_template
            
            # Minor 7th (1, b3, 5, b7)
            min7_template = np.zeros(12)
            min7_template[root_idx] = 1.0                         # Root
            min7_template[(root_idx + 3) % 12] = 0.8              # Minor third
            min7_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            min7_template[(root_idx + 10) % 12] = 0.7             # Minor seventh
            self.chord_templates[f"{root_note}m7"] = min7_template
            
            # Suspended 4th (1, 4, 5)
            sus4_template = np.zeros(12)
            sus4_template[root_idx] = 1.0                         # Root
            sus4_template[(root_idx + 5) % 12] = 0.8              # Perfect fourth
            sus4_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            self.chord_templates[f"{root_note}sus4"] = sus4_template
            
            # Suspended 2nd (1, 2, 5)
            sus2_template = np.zeros(12)
            sus2_template[root_idx] = 1.0                         # Root
            sus2_template[(root_idx + 2) % 12] = 0.8              # Major second
            sus2_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            self.chord_templates[f"{root_note}sus2"] = sus2_template
            
            # Augmented (1, 3, #5)
            aug_template = np.zeros(12)
            aug_template[root_idx] = 1.0                          # Root
            aug_template[(root_idx + 4) % 12] = 0.8               # Major third
            aug_template[(root_idx + 8) % 12] = 0.8               # Augmented fifth
            self.chord_templates[f"{root_note}aug"] = aug_template
            
            # Diminished (1, b3, b5)
            dim_template = np.zeros(12)
            dim_template[root_idx] = 1.0                          # Root
            dim_template[(root_idx + 3) % 12] = 0.8               # Minor third
            dim_template[(root_idx + 6) % 12] = 0.8               # Diminished fifth
            self.chord_templates[f"{root_note}dim"] = dim_template
            
            # Add9 (1, 3, 5, 9)
            add9_template = np.zeros(12)
            add9_template[root_idx] = 1.0                         # Root
            add9_template[(root_idx + 4) % 12] = 0.8              # Major third
            add9_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            add9_template[(root_idx + 2) % 12] = 0.6              # Major ninth
            self.chord_templates[f"{root_note}add9"] = add9_template
            
            # 9th chord (1, 3, 5, b7, 9)
            ninth_template = np.zeros(12)
            ninth_template[root_idx] = 1.0                        # Root
            ninth_template[(root_idx + 4) % 12] = 0.8             # Major third
            ninth_template[(root_idx + 7) % 12] = 0.8             # Perfect fifth
            ninth_template[(root_idx + 10) % 12] = 0.7            # Minor seventh
            ninth_template[(root_idx + 2) % 12] = 0.6             # Major ninth
            self.chord_templates[f"{root_note}9"] = ninth_template
            
            # Minor 9th (1, b3, 5, b7, 9)
            min9_template = np.zeros(12)
            min9_template[root_idx] = 1.0                         # Root
            min9_template[(root_idx + 3) % 12] = 0.8              # Minor third
            min9_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            min9_template[(root_idx + 10) % 12] = 0.7             # Minor seventh
            min9_template[(root_idx + 2) % 12] = 0.6              # Major ninth
            self.chord_templates[f"{root_note}m9"] = min9_template
            
            # 6th chord (1, 3, 5, 6)
            sixth_template = np.zeros(12)
            sixth_template[root_idx] = 1.0                        # Root
            sixth_template[(root_idx + 4) % 12] = 0.8             # Major third
            sixth_template[(root_idx + 7) % 12] = 0.8             # Perfect fifth
            sixth_template[(root_idx + 9) % 12] = 0.7             # Major sixth
            self.chord_templates[f"{root_note}6"] = sixth_template
            
            # Minor 6th (1, b3, 5, 6)
            min6_template = np.zeros(12)
            min6_template[root_idx] = 1.0                         # Root
            min6_template[(root_idx + 3) % 12] = 0.8              # Minor third
            min6_template[(root_idx + 7) % 12] = 0.8              # Perfect fifth
            min6_template[(root_idx + 9) % 12] = 0.7              # Major sixth
            self.chord_templates[f"{root_note}m6"] = min6_template
            
        # Power chord (just root and fifth)
        for root_idx, root_note in enumerate(self.NOTES):
            power_template = np.zeros(12)
            power_template[root_idx] = 1.0                        # Root
            power_template[(root_idx + 7) % 12] = 1.0             # Perfect fifth
            self.chord_templates[f"{root_note}5"] = power_template
            
        print(f"Loaded {len(self.chord_templates)} chord templates")
        
    def load_audio(self, file_path):
        """Load an audio file for analysis."""
        try:
            y, sr = librosa.load(file_path)
            self.audio = y
            self.sr = sr
            self.duration = librosa.get_duration(y=y, sr=sr)
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def extract_features(self):
        """Extract chromagram and additional features from the audio."""
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please load an audio file first.")
            return None
        
        # Compute chromagram using CQT (Constant-Q Transform) for better pitch representation
        self.chroma_cqt = librosa.feature.chroma_cqt(
            y=self.audio, 
            sr=self.sr,
            hop_length=self.hop_length,
            bins_per_octave=36,  # Higher resolution for better note distinction
            n_octaves=7          # Cover a wide pitch range
        )
        
        # Also compute chromagram using STFT for comparison
        self.chroma_stft = librosa.feature.chroma_stft(
            y=self.audio, 
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Combine both chromagrams with more weight to CQT
        self.chroma = 0.75 * self.chroma_cqt + 0.25 * self.chroma_stft
        
        # Apply smoothing to reduce noise
        self.chroma_smooth = np.copy(self.chroma)
        for i in range(12):
            self.chroma_smooth[i] = gaussian_filter1d(self.chroma[i], sigma=2)
        
        # Normalize the chromagram
        self.chroma_norm = librosa.util.normalize(self.chroma_smooth, axis=0)
        
        # Get timing information
        self.times = librosa.times_like(self.chroma, sr=self.sr, hop_length=self.hop_length)
        
        # Extract bass chromagram (focused on lower frequencies)
        # This helps in detecting bass notes for slash chords
        self.bass_chroma = librosa.feature.chroma_cqt(
            y=self.audio, 
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=librosa.note_to_hz('C1'),  # Focus on bass range
            n_octaves=3                      # Only include lower octaves
        )
        
        # Smooth and normalize bass chromagram
        self.bass_chroma_smooth = np.copy(self.bass_chroma)
        for i in range(12):
            self.bass_chroma_smooth[i] = gaussian_filter1d(self.bass_chroma[i], sigma=3)
        self.bass_chroma_norm = librosa.util.normalize(self.bass_chroma_smooth, axis=0)
        
        # Detect onsets for segmentation
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr, hop_length=self.hop_length)
        self.onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=self.sr, 
            hop_length=self.hop_length,
            units='time'
        )
        
        print(f"Extracted chromagram with shape: {self.chroma.shape}")
        return self.chroma_norm
    
    def detect_bass_note(self, bass_chroma_frame):
        """Detect the most prominent bass note in a frame."""
        # Find the index of the maximum value
        max_idx = np.argmax(bass_chroma_frame)
        # Only return if the value is significant
        if bass_chroma_frame[max_idx] > 0.7:
            return self.NOTES[max_idx]
        return None
    
    def recognize_chords(self, segmentation='adaptive', min_duration=0.25):
        """
        Recognize chords from the chromagram using template matching.
        
        Parameters:
        - segmentation: Method for segmenting audio ('fixed', 'onset', or 'adaptive')
        - min_duration: Minimum duration for a chord segment (in seconds)
        """
        if not hasattr(self, 'chroma_norm'):
            print("No features extracted. Please extract features first.")
            return None
        
        # Determine segments based on the chosen method
        segments = []
        
        if segmentation == 'fixed':
            # Simple fixed-size segmentation
            window_frames = max(1, int(min_duration * self.sr / self.hop_length))
            for i in range(0, self.chroma_norm.shape[1], window_frames):
                end_idx = min(i + window_frames, self.chroma_norm.shape[1])
                segments.append((i, end_idx))
                
        elif segmentation == 'onset':
            # Use onset detection for segmentation
            if len(self.onsets) > 0:
                onset_frames = librosa.time_to_frames(self.onsets, sr=self.sr, hop_length=self.hop_length)
                prev_frame = 0
                for frame in onset_frames:
                    # Ensure minimum segment length
                    if frame - prev_frame >= int(min_duration * self.sr / self.hop_length):
                        segments.append((prev_frame, frame))
                        prev_frame = frame
                # Add final segment
                if self.chroma_norm.shape[1] - prev_frame > 0:
                    segments.append((prev_frame, self.chroma_norm.shape[1]))
            else:
                # Fallback to fixed segmentation if no onsets found
                window_frames = max(1, int(min_duration * self.sr / self.hop_length))
                for i in range(0, self.chroma_norm.shape[1], window_frames):
                    end_idx = min(i + window_frames, self.chroma_norm.shape[1])
                    segments.append((i, end_idx))
        
        elif segmentation == 'adaptive':
            # Adaptive segmentation using both onsets and chord changes
            # Start with onset-based segmentation
            if len(self.onsets) > 0:
                onset_frames = librosa.time_to_frames(self.onsets, sr=self.sr, hop_length=self.hop_length)
                prev_frame = 0
                for frame in onset_frames:
                    # Ensure minimum segment length
                    if frame - prev_frame >= int(min_duration * self.sr / self.hop_length):
                        segments.append((prev_frame, frame))
                        prev_frame = frame
                # Add final segment
                if self.chroma_norm.shape[1] - prev_frame > 0:
                    segments.append((prev_frame, self.chroma_norm.shape[1]))
            else:
                # Fallback to fixed segmentation if no onsets found
                window_frames = max(1, int(min_duration * self.sr / self.hop_length))
                for i in range(0, self.chroma_norm.shape[1], window_frames):
                    end_idx = min(i + window_frames, self.chroma_norm.shape[1])
                    segments.append((i, end_idx))
                
            # Further segmentation will happen during chord detection
        
        # Initialize results
        chord_timeline = []
        
        # Process each segment
        for start_frame, end_frame in segments:
            # Calculate average chroma vector for the segment
            segment_chroma = np.mean(self.chroma_norm[:, start_frame:end_frame], axis=1)
            segment_bass_chroma = np.mean(self.bass_chroma_norm[:, start_frame:end_frame], axis=1)
            
            # Detect bass note
            bass_note = self.detect_bass_note(segment_bass_chroma)
            
            # Compare with chord templates
            similarities = {}
            for chord_name, template in self.chord_templates.items():
                # Calculate similarity (correlation coefficient)
                similarity = np.corrcoef(segment_chroma, template)[0, 1]
                if not np.isnan(similarity):  # Handle NaN values
                    similarities[chord_name] = similarity
            
            # Get the best matches
            if similarities:
                # Sort by similarity score
                sorted_chords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                best_chord, best_similarity = sorted_chords[0]
                
                # Only accept if similarity is high enough
                if best_similarity > 0.5:
                    # Check for slash chord (if bass note detected and different from root)
                    chord_root = best_chord[0] if len(best_chord) == 1 else best_chord[0:2] if best_chord[1] == '#' else best_chord[0]
                    
                    # Format the chord name
                    chord_name = best_chord
                    if bass_note and bass_note != chord_root:
                        chord_name = f"{best_chord}/{bass_note}"
                    
                    # Record chord information
                    start_time = self.times[start_frame]
                    end_time = self.times[min(end_frame - 1, len(self.times) - 1)]
                    
                    chord_timeline.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'chord': chord_name,
                        'confidence': best_similarity,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
        
        # Adaptive post-processing: combine identical adjacent chords
        if len(chord_timeline) > 1:
            processed_timeline = [chord_timeline[0]]
            for i in range(1, len(chord_timeline)):
                current = chord_timeline[i]
                previous = processed_timeline[-1]
                
                # If same chord, extend the previous entry
                if current['chord'] == previous['chord']:
                    previous['end_time'] = current['end_time']
                    previous['end_frame'] = current['end_frame']
                    # Update confidence (weighted average)
                    prev_duration = previous['end_time'] - previous['start_time']
                    curr_duration = current['end_time'] - current['start_time']
                    total_duration = prev_duration + curr_duration
                    previous['confidence'] = (
                        (prev_duration * previous['confidence'] + 
                         curr_duration * current['confidence']) / total_duration
                    )
                else:
                    processed_timeline.append(current)
            
            self.chord_timeline = processed_timeline
        else:
            self.chord_timeline = chord_timeline
        
        return self.chord_timeline
    
    def analyze_progression(self):
        """Analyze the chord progression to find patterns and the key."""
        if not hasattr(self, 'chord_timeline') or len(self.chord_timeline) == 0:
            print("No chord timeline available. Please recognize chords first.")
            return None
        
        # Extract the sequence of chords
        chord_sequence = [item['chord'] for item in self.chord_timeline]
        
        # Clean chord names for analysis (remove bass notes)
        clean_chords = []
        for chord in chord_sequence:
            if '/' in chord:
                # Remove slash chord bass note
                clean_chords.append(chord.split('/')[0])
            else:
                clean_chords.append(chord)
        
        # Count occurrences of each chord
        chord_counts = {}
        for chord in clean_chords:
            if chord in chord_counts:
                chord_counts[chord] += 1
            else:
                chord_counts[chord] = 1
        
        # Find the most common chord (could be the tonic)
        most_common_chord = max(chord_counts.items(), key=lambda x: x[1])[0]
        
        # Extract the root of the most common chord
        if len(most_common_chord) > 0:
            if len(most_common_chord) > 1 and most_common_chord[1] == '#':
                root_note = most_common_chord[0:2]
                chord_type = most_common_chord[2:]
            else:
                root_note = most_common_chord[0]
                chord_type = most_common_chord[1:]
        else:
            root_note = ''
            chord_type = ''
        
        # Key detection based on chord prevalence and music theory
        major_keys = self._score_keys(clean_chords, mode='major')
        minor_keys = self._score_keys(clean_chords, mode='minor')
        
        # Combine results
        key_scores = {**major_keys, **minor_keys}
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top key candidates
        top_keys = sorted_keys[:3]
        
        # Find common chord progressions
        progressions = self._find_progressions(clean_chords)
        
        # Return analysis results
        analysis = {
            'most_common_chord': most_common_chord,
            'chord_counts': chord_counts,
            'possible_keys': top_keys,
            'common_progressions': progressions[:5] if len(progressions) > 5 else progressions
        }
        
        self.progression_analysis = analysis
        
        # Print key analysis
        print("\nKey Analysis:")
        for key, score in top_keys:
            print(f"{key}: Score {score:.2f}")
        
        # Print most common chord
        print(f"\nMost common chord: {most_common_chord}")
        
        # Print common progressions
        print("\nCommon chord progressions:")
        for prog, count in progressions[:5] if len(progressions) > 5 else progressions:
            progression_str = " → ".join(prog)
            print(f"{progression_str}: {count} occurrences")
        
        return analysis
    
    def _score_keys(self, chord_sequence, mode='major'):
        """Score possible keys based on chords present."""
        key_scores = {}
        
        # Define diatonic chords for each key
        for root_idx, root in enumerate(self.NOTES):
            if mode == 'major':
                key_name = f"{root} major"
                # Diatonic chords in major keys (I, ii, iii, IV, V, vi, vii°)
                diatonic = [
                    root,                                              # I  (major)
                    f"{self.NOTES[(root_idx + 2) % 12]}m",            # ii (minor)
                    f"{self.NOTES[(root_idx + 4) % 12]}m",            # iii (minor)
                    f"{self.NOTES[(root_idx + 5) % 12]}",             # IV (major)
                    f"{self.NOTES[(root_idx + 7) % 12]}",             # V  (major)
                    f"{self.NOTES[(root_idx + 9) % 12]}m",            # vi (minor)
                    f"{self.NOTES[(root_idx + 11) % 12]}dim"          # vii° (diminished)
                ]
                
                # Common extensions and substitutions
                extensions = [
                    f"{root}7", f"{root}maj7", f"{root}6", f"{root}add9",  # I extensions
                    f"{self.NOTES[(root_idx + 2) % 12]}m7",                # ii extensions
                    f"{self.NOTES[(root_idx + 4) % 12]}m7",                # iii extensions 
                    f"{self.NOTES[(root_idx + 5) % 12]}7", f"{self.NOTES[(root_idx + 5) % 12]}maj7",  # IV extensions
                    f"{self.NOTES[(root_idx + 7) % 12]}7", f"{self.NOTES[(root_idx + 7) % 12]}9",     # V extensions
                    f"{self.NOTES[(root_idx + 9) % 12]}m7",                # vi extensions
                    f"{self.NOTES[(root_idx + 11) % 12]}m7b5"              # vii extensions
                ]
                
                # Secondary dominants
                secondary = [
                    f"{self.NOTES[(root_idx + 2) % 12]}7",   # V/ii
                    f"{self.NOTES[(root_idx + 4) % 12]}7",   # V/iii
                    f"{self.NOTES[(root_idx + 5) % 12]}7",   # V/IV
                    f"{self.NOTES[(root_idx + 7) % 12]}7",   # V/V
                    f"{self.NOTES[(root_idx + 9) % 12]}7",   # V/vi
                ]
                
            elif mode == 'minor':
                key_name = f"{root} minor"
                # Diatonic chords in natural minor keys (i, ii°, III, iv, v, VI, VII)
                diatonic = [
                    f"{root}m",                                        # i  (minor)
                    f"{self.NOTES[(root_idx + 2) % 12]}dim",          # ii° (diminished)
                    f"{self.NOTES[(root_idx + 3) % 12]}",             # III (major)
                    f"{self.NOTES[(root_idx + 5) % 12]}m",            # iv (minor)
                    f"{self.NOTES[(root_idx + 7) % 12]}m",            # v  (minor)
                    f"{self.NOTES[(root_idx + 8) % 12]}",             # VI (major)
                    f"{self.NOTES[(root_idx + 10) % 12]}"             # VII (major)
                ]
                
                # Harmonic minor additions (V, vii°)
                harmonic_additions = [
                    f"{self.NOTES[(root_idx + 7) % 12]}",             # V  (major from harmonic minor)
                    f"{self.NOTES[(root_idx + 11) % 12]}dim"          # vii° (diminished from harmonic minor)
                ]
                
                # Extensions and 7th chords
                extensions = [
                    f"{root}m7", f"{root}m6", f"{root}m9",                  # i extensions
                    f"{self.NOTES[(root_idx + 3) % 12]}7", f"{self.NOTES[(root_idx + 3) % 12]}maj7",  # III extensions
                    f"{self.NOTES[(root_idx + 5) % 12]}m7",                 # iv extensions
                    f"{self.NOTES[(root_idx + 7) % 12]}7",                  # V extensions (harmonic minor)
                    f"{self.NOTES[(root_idx + 8) % 12]}7", f"{self.NOTES[(root_idx + 8) % 12]}maj7",  # VI extensions
                    f"{self.NOTES[(root_idx + 10) % 12]}7"                  # VII extensions
                ]
                
                secondary = [
                    f"{self.NOTES[(root_idx + 3) % 12]}7",   # V/III
                    f"{self.NOTES[(root_idx + 5) % 12]}7",   # V/iv
                    f"{self.NOTES[(root_idx + 7) % 12]}7",   # V/v
                    f"{self.NOTES[(root_idx + 8) % 12]}7",   # V/VI
                ]
                
                diatonic.extend(harmonic_additions)
            
            # Calculate score based on chord frequencies
            score = 0
            for chord in chord_sequence:
                # Strip extensions for basic matching (C7 -> C, Am7 -> Am, etc.)
                basic_chord = chord.split('7')[0].split('9')[0].split('maj')[0].split('dim')[0].split('aug')[0].split('sus')[0]
                if basic_chord in diatonic:
                    score += 1.0  # Full point for diatonic chords
                elif chord in extensions:
                    score += 0.8  # Points for extended diatonic chords
                elif chord in secondary:
                    score += 0.6  # Points for secondary dominants
                elif basic_chord in extensions or basic_chord in secondary:
                    score += 0.5  # Points for basic version of extensions
            
            # Normalize score by sequence length
            if chord_sequence:
                score /= len(chord_sequence)
                key_scores[key_name] = score
        
        return key_scores
    
    def _find_progressions(self, chord_sequence, min_length=2, max_length=4):
        """Find common chord progressions in the sequence."""
        progression_counts = {}
        
        # Look for progressions of different lengths
        for length in range(min_length, max_length + 1):
            for i in range(len(chord_sequence) - length + 1):
                progression = tuple(chord_sequence[i:i+length])
                if progression in progression_counts:
                    progression_counts[progression] += 1
                else:
                    progression_counts[progression] = 1
        
        # Sort progressions by count
        sorted_progressions = sorted(progression_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_progressions
    
    def export_to_json(self, filename="chord_analysis.json"):
        """Export the analysis results to a JSON file."""
        if not hasattr(self, 'chord_timeline') or not hasattr(self, 'progression_analysis'):
            print("Missing analysis data. Please run chord recognition and progression analysis first.")
            return False
        
        # Prepare data for export
        export_data = {
            'chords': self.chord_timeline,
            'progression_analysis': self.progression_analysis
        }
        
        # Convert to JSON
        try:
            # Convert NumPy data types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            # Convert all data to JSON-serializable types
            serializable_data = convert_numpy_types(export_data)
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"Analysis exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def visualize_chords(self, show_bass=True, show_confidence=True):
        """Visualize the chroma features and detected chords."""
        if not hasattr(self, 'chroma_norm') or not hasattr(self, 'chord_timeline'):
            print("Missing data for visualization. Please extract features and recognize chords first.")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Plot chromagram
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        # Store the returned mappable object for the colorbar
        img = librosa.display.specshow(self.chroma_norm, y_axis='chroma', x_axis='time', 
                                hop_length=self.hop_length, sr=self.sr, ax=ax1)
        ax1.set_title('Chromagram')
        plt.colorbar(mappable=img, ax=ax1)  # Pass the mappable explicitly
        
        # Plot bass chromagram if requested
        if show_bass and hasattr(self, 'bass_chroma_norm'):
            ax2 = plt.subplot2grid((3, 1), (1, 0))
            img2 = librosa.display.specshow(self.bass_chroma_norm, y_axis='chroma', x_axis='time', 
                                    hop_length=self.hop_length, sr=self.sr, ax=ax2)
            ax2.set_title('Bass Chromagram')
            plt.colorbar(mappable=img2, ax=ax2)  # Pass the mappable explicitly
            
            # Adjust subplot position for chord timeline
            ax3 = plt.subplot2grid((3, 1), (2, 0))
        else:
            # Use more space for chord timeline if no bass plot
            ax3 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        
        # Create a list of unique chords for y-axis
        unique_chords = list(set(item['chord'] for item in self.chord_timeline))
        unique_chords.sort()
        
        # Create a chord map
        chord_indices = {chord: i for i, chord in enumerate(unique_chords)}
        
        # Plot each chord segment
        for item in self.chord_timeline:
            chord = item['chord']
            start = item['start_time']
            end = item['end_time']
            confidence = item['confidence']
            
            # Adjust alpha based on confidence if requested
            alpha = min(1.0, confidence) if show_confidence else 0.8
            
            # Calculate color based on chord type
            if 'm' in chord and not 'maj' in chord:  # Minor chords
                color = 'blue'
            elif 'dim' in chord:  # Diminished chords
                color = 'purple'
            elif 'aug' in chord:  # Augmented chords
                color = 'red'
            elif '7' in chord or '9' in chord:  # Seventh or ninth chords
                color = 'green'
            elif 'sus' in chord:  # Suspended chords
                color = 'orange'
            else:  # Major chords and others
                color = 'teal'
            
            # Plot horizontal line for chord duration
            ax3.hlines(chord_indices[chord], start, end, color=color, alpha=alpha, linewidth=5)
            
            # Add label in middle of segment if long enough
            if end - start > 0.5:  # Only label segments longer than 0.5 seconds
                # Add confidence display if requested
                if show_confidence:
                    label = f"{chord} ({confidence:.2f})"
                else:
                    label = chord
                ax3.text((start + end) / 2, chord_indices[chord], label, 
                        fontsize=8, ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.6))
        
        ax3.set_yticks(range(len(unique_chords)))
        ax3.set_yticklabels(unique_chords)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Chord')
        ax3.set_title('Detected Chord Progression')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_chord_analysis.png')
        print("Visualization saved as 'advanced_chord_analysis.png'")
        return plt.gcf()  # Return the current figure
        
    def suggest_scales(self):
        """Suggest scales that would work with the detected chords."""
        if not hasattr(self, 'progression_analysis'):
            print("Missing progression analysis. Please run analyze_progression() first.")
            return None
        
        # Get top key candidates
        possible_keys = self.progression_analysis['possible_keys']
        
        # Define scale patterns
        scale_patterns = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'natural_minor': [0, 2, 3, 5, 7, 8, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'pentatonic_major': [0, 2, 4, 7, 9],
            'pentatonic_minor': [0, 3, 5, 7, 10],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        
        suggested_scales = []
        
        for key_name, score in possible_keys:
            # Parse key name
            parts = key_name.split()
            if len(parts) == 2:
                root, mode = parts
                
                # Generate scales based on key
                root_idx = self.NOTES.index(root)
                
                if mode == 'major':
                    # For major keys, suggest major and related scales
                    scales = [
                        {'name': f"{root} Major (Ionian)", 'notes': self._get_scale_notes(root_idx, scale_patterns['major'])},
                        {'name': f"{root} Lydian", 'notes': self._get_scale_notes(root_idx, scale_patterns['lydian'])},
                        {'name': f"{root} Mixolydian", 'notes': self._get_scale_notes(root_idx, scale_patterns['mixolydian'])},
                        {'name': f"{root} Major Pentatonic", 'notes': self._get_scale_notes(root_idx, scale_patterns['pentatonic_major'])}
                    ]
                    
                    # Also suggest relative minor
                    relative_minor_idx = (root_idx + 9) % 12
                    relative_minor = self.NOTES[relative_minor_idx]
                    scales.append({'name': f"{relative_minor} Natural Minor", 
                                  'notes': self._get_scale_notes(relative_minor_idx, scale_patterns['natural_minor'])})
                    
                elif mode == 'minor':
                    # For minor keys, suggest minor and related scales
                    scales = [
                        {'name': f"{root} Natural Minor (Aeolian)", 'notes': self._get_scale_notes(root_idx, scale_patterns['natural_minor'])},
                        {'name': f"{root} Harmonic Minor", 'notes': self._get_scale_notes(root_idx, scale_patterns['harmonic_minor'])},
                        {'name': f"{root} Melodic Minor", 'notes': self._get_scale_notes(root_idx, scale_patterns['melodic_minor'])},
                        {'name': f"{root} Dorian", 'notes': self._get_scale_notes(root_idx, scale_patterns['dorian'])},
                        {'name': f"{root} Phrygian", 'notes': self._get_scale_notes(root_idx, scale_patterns['phrygian'])},
                        {'name': f"{root} Minor Pentatonic", 'notes': self._get_scale_notes(root_idx, scale_patterns['pentatonic_minor'])},
                        {'name': f"{root} Blues", 'notes': self._get_scale_notes(root_idx, scale_patterns['blues'])}
                    ]
                
                # Add scales to suggestions with their key score
                for scale in scales:
                    suggested_scales.append({
                        'scale': scale['name'],
                        'notes': scale['notes'],
                        'key_score': score
                    })
        
        # Sort scales by key score
        suggested_scales.sort(key=lambda x: x['key_score'], reverse=True)
        
        # Take top 5 suggestions
        top_suggestions = suggested_scales[:5]
        
        # Print suggestions
        print("\nSuggested scales for soloing/improvisation:")
        for i, suggestion in enumerate(top_suggestions):
            print(f"{i+1}. {suggestion['scale']}")
            print(f"   Notes: {', '.join(suggestion['notes'])}")
            print(f"   Key certainty: {suggestion['key_score']:.2f}")
        
        return top_suggestions
    
    def _get_scale_notes(self, root_idx, pattern):
        """Generate scale notes from a root note and pattern."""
        return [self.NOTES[(root_idx + interval) % 12] for interval in pattern]


def main():
    """Main function to demonstrate the advanced chord analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python advanced_chord_analyzer.py <audio_file> [--segmentation=adaptive|fixed|onset]")
        return
    
    audio_file = sys.argv[1]
    
    # Parse optional arguments
    segmentation = 'adaptive'  # Default
    for arg in sys.argv[2:]:
        if arg.startswith('--segmentation='):
            segmentation = arg.split('=')[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found.")
        return
    
    # Create analyzer and process the file
    analyzer = AdvancedChordAnalyzer()
    if analyzer.load_audio(audio_file):
        print("\nExtracting audio features...")
        analyzer.extract_features()
        
        print(f"\nDetecting chords using {segmentation} segmentation...")
        chord_timeline = analyzer.recognize_chords(segmentation=segmentation)
        
        if chord_timeline:
            print("\nDetected Chords:")
            for item in chord_timeline:
                print(f"{item['start_time']:.2f}s - {item['end_time']:.2f}s: {item['chord']} (confidence: {item['confidence']:.2f})")
            
            print("\nAnalyzing chord progression...")
            analyzer.analyze_progression()
            
            print("\nSuggesting scales for improvisation...")
            analyzer.suggest_scales()
            
            print("\nGenerating visualization...")
            analyzer.visualize_chords()
            
            print("\nExporting analysis to JSON...")
            analyzer.export_to_json()
        else:
            print("No chords detected.")

if __name__ == "__main__":
    main()