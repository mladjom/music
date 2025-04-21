#!/usr/bin/env python3
# Advanced Lyrics Extraction and Alignment Module
# For use with the Advanced Track Separator

import os
import sys
import time
import json
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class LyricsExtractor:
    """
    Advanced module for extracting lyrics from vocal tracks and
    aligning them with the music timeline.
    
    Features:
    - High-quality transcription using Whisper models
    - Time-aligned lyrics with timestamps
    - Word-level alignment for karaoke-style applications
    - Multiple language support
    - Confidence scoring for each detected segment
    """
    
    def __init__(self, device=None, model_size="medium"):
        """Initialize the lyrics extractor."""
        # Set device (CPU or GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model parameters
        self.model_size = model_size  # base, small, medium, large
        self.model = None
        self.processor = None
        
        # State variables
        self.audio = None
        self.sr = None
        self.lyrics = None
        self.lyric_segments = None
        self.word_alignments = None
        self.language = None
        
        # Check if whisper is available
        try:
            import whisper
            self.whisper_available = True
            print("Whisper is available for transcription")
        except ImportError:
            self.whisper_available = False
            print("Warning: Whisper not found. Please install it with: pip install openai-whisper")
    
    def load_model(self):
        """Load the Whisper model for transcription."""
        if not self.whisper_available:
            print("Error: Whisper is required but not available.")
            return False
        
        try:
            import whisper
            print(f"Loading Whisper model (size: {self.model_size})...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Model loaded: {self.model.dims.n_text_state}m params")
            return True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return False
    
    def load_audio(self, file_path):
        """Load an audio file containing vocals for transcription."""
        try:
            # Use librosa for loading (handles more formats)
            self.audio, self.sr = librosa.load(file_path, sr=16000, mono=True)  # Whisper expects 16kHz
            self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
            
            print(f"Loaded vocal audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def transcribe(self, language=None, task="transcribe"):
        """
        Transcribe the loaded vocal audio to extract lyrics.
        
        Parameters:
        - language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
        - task: 'transcribe' or 'translate' (to English)
        """
        if self.audio is None:
            print("No audio loaded. Please load an audio file first.")
            return False
        
        if self.model is None:
            if not self.load_model():
                return False
        
        try:
            print(f"Transcribing vocals{'in ' + language if language else ''}...")
            transcription = self.model.transcribe(
                self.audio,
                language=language,
                task=task,
                word_timestamps=True  # Get timestamps for individual words when possible
            )
            
            # Store results
            self.lyrics = transcription["text"]
            self.language = transcription["language"]
            self.lyric_segments = transcription["segments"]
            
            # Extract word-level alignments when available
            self.word_alignments = []
            for segment in self.lyric_segments:
                if "words" in segment:
                    self.word_alignments.extend(segment["words"])
            
            print(f"Transcription complete - detected language: {self.language}")
            print(f"Found {len(self.lyric_segments)} lyric segments")
            print(f"Found {len(self.word_alignments)} aligned words")
            
            return True
        except Exception as e:
            print(f"Error during transcription: {e}")
            return False
    
    def refine_alignments(self):
        """
        Refine the timing of lyric alignments using audio analysis.
        This helps improve accuracy for musical performances where
        speech-to-text models might not perfectly align with sung vocals.
        """
        if not self.lyric_segments:
            print("No lyrics transcribed yet. Please run transcribe() first.")
            return False
        
        try:
            print("Refining lyric alignments...")
            
            # Detect onsets in the vocal track
            onset_env = librosa.onset.onset_strength(
                y=self.audio, 
                sr=self.sr,
                hop_length=512,
                aggregate=np.median  # More suitable for vocals
            )
            
            # Find onset frames
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, 
                sr=self.sr,
                hop_length=512,
                units='time'
            )
            
            # For each segment, find nearby onsets that might represent 
            # a better alignment point for the start of the segment
            for i, segment in enumerate(self.lyric_segments):
                segment_start = segment["start"]
                
                # Find the closest onset to this segment's start time
                closest_onset_idx = np.argmin(np.abs(onsets - segment_start))
                closest_onset = onsets[closest_onset_idx]
                
                # If the onset is within a reasonable threshold, adjust the segment
                if abs(closest_onset - segment_start) < 0.3:  # 300ms threshold
                    # Store original for reference
                    segment["original_start"] = segment_start
                    # Update with improved timing
                    segment["start"] = closest_onset
                    
                    # Adjust end time of previous segment if needed
                    if i > 0 and self.lyric_segments[i-1]["end"] > closest_onset:
                        self.lyric_segments[i-1]["end"] = closest_onset
            
            print("Alignment refinement complete")
            return True
        except Exception as e:
            print(f"Error during alignment refinement: {e}")
            return False
    
    def generate_karaoke_lyrics(self, max_words_per_line=7):
        """
        Generate properly formatted karaoke-style lyrics with timestamps.
        
        Parameters:
        - max_words_per_line: Maximum number of words per line for better readability
        
        Returns a list of timed lyric lines suitable for karaoke display.
        """
        if not self.lyric_segments:
            print("No lyrics transcribed yet. Please run transcribe() first.")
            return []
        
        karaoke_lines = []
        current_line = {"words": [], "start": 0, "end": 0}
        word_count = 0
        
        # If we have word-level alignments, use those for precise timing
        if self.word_alignments:
            for word_info in self.word_alignments:
                word = word_info["word"]
                # Skip leading space in word (Whisper artifact)
                if word.startswith(" "):
                    word = word[1:]
                    
                # Skip empty words
                if not word:
                    continue
                
                # If this is the first word in the line, set the start time
                if word_count == 0:
                    current_line["start"] = word_info["start"]
                
                # Add the word to the current line
                current_line["words"].append({
                    "word": word,
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
                word_count += 1
                
                # End of line detection: max words reached or punctuation
                line_break = False
                if word_count >= max_words_per_line:
                    line_break = True
                elif word.endswith((".", "!", "?", ":", ";", ",")):
                    line_break = True
                
                if line_break:
                    # Set the end time to the last word's end time
                    current_line["end"] = word_info["end"]
                    # Create the full text by joining words
                    current_line["text"] = " ".join([w["word"] for w in current_line["words"]])
                    
                    # Add the line to our results
                    karaoke_lines.append(current_line)
                    
                    # Reset for the next line
                    current_line = {"words": [], "start": 0, "end": 0}
                    word_count = 0
            
            # Add any remaining words as the last line
            if current_line["words"]:
                current_line["end"] = current_line["words"][-1]["end"]
                current_line["text"] = " ".join([w["word"] for w in current_line["words"]])
                karaoke_lines.append(current_line)
                
        else:
            # Fall back to segment-level if word alignments aren't available
            for segment in self.lyric_segments:
                # Split the segment text into words
                words = segment["text"].split()
                
                while words:
                    # Take a chunk of words up to max_words_per_line
                    chunk = words[:max_words_per_line]
                    words = words[max_words_per_line:]
                    
                    # Calculate timing (evenly distribute across the segment)
                    chunk_duration = segment["end"] - segment["start"]
                    words_per_second = len(chunk) / chunk_duration if chunk_duration > 0 else 1
                    
                    start_time = segment["start"]
                    line_words = []
                    
                    for i, word in enumerate(chunk):
                        word_duration = 1 / words_per_second
                        word_start = start_time + i / words_per_second
                        word_end = word_start + word_duration
                        
                        line_words.append({
                            "word": word,
                            "start": word_start,
                            "end": word_end
                        })
                    
                    karaoke_line = {
                        "words": line_words,
                        "text": " ".join(chunk),
                        "start": start_time,
                        "end": start_time + len(chunk) / words_per_second
                    }
                    
                    karaoke_lines.append(karaoke_line)
        
        print(f"Generated {len(karaoke_lines)} karaoke lines")
        return karaoke_lines
    
    def export_lyrics(self, output_dir="."):
        """Export lyrics and alignments to various formats."""
        if not self.lyrics:
            print("No lyrics available. Please run transcribe() first.")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = "lyrics"
        
        # 1. Plain text lyrics
        with open(os.path.join(output_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(self.lyrics)
        
        # 2. JSON with segments and timing information
        full_data = {
            "text": self.lyrics,
            "language": self.language,
            "segments": self.lyric_segments,
            "word_alignments": self.word_alignments
        }
        
        with open(os.path.join(output_dir, f"{base_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2)
        
        # 3. SRT subtitle format (for video)
        self._export_srt(os.path.join(output_dir, f"{base_name}.srt"))
        
        # 4. LRC format (for music players)
        self._export_lrc(os.path.join(output_dir, f"{base_name}.lrc"))
        
        # 5. Karaoke format
        karaoke_lyrics = self.generate_karaoke_lyrics()
        with open(os.path.join(output_dir, f"{base_name}_karaoke.json"), 'w', encoding='utf-8') as f:
            json.dump(karaoke_lyrics, f, indent=2)
        
        print(f"Lyrics exported to {output_dir} in multiple formats")
        return True
    
    def _export_srt(self, filename):
        """Export lyrics in SRT subtitle format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.lyric_segments):
                # Convert timestamps to SRT format (HH:MM:SS,mmm)
                start_time = self._format_timestamp_srt(segment["start"])
                end_time = self._format_timestamp_srt(segment["end"])
                
                # Write SRT entry
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
    
    def _export_lrc(self, filename):
        """Export lyrics in LRC format for music players."""
        with open(filename, 'w', encoding='utf-8') as f:
            # Add LRC header metadata
            f.write("[ar:Unknown Artist]\n")
            f.write("[ti:Unknown Title]\n")
            f.write("[length:{}]\n".format(self._format_timestamp_lrc(self.duration)))
            
            # Write time-synced lyrics
            for segment in self.lyric_segments:
                timestamp = self._format_timestamp_lrc(segment["start"])
                f.write(f"[{timestamp}]{segment['text'].strip()}\n")
    
    def _format_timestamp_srt(self, seconds):
        """Format a timestamp for SRT format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    
    def _format_timestamp_lrc(self, seconds):
        """Format a timestamp for LRC format: mm:ss.xx"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    
    def visualize_alignments(self, output_file="lyrics_alignment.png"):
        """Visualize the lyrics alignments with the audio waveform."""
        if not self.lyric_segments:
            print("No lyrics transcribed yet. Please run transcribe() first.")
            return False
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(self.audio, sr=self.sr)
            plt.title("Vocal Waveform with Lyric Segments")
            
            # Mark segment boundaries
            for segment in self.lyric_segments:
                plt.axvline(x=segment["start"], color='r', linestyle='--', alpha=0.7)
                plt.axvline(x=segment["end"], color='b', linestyle=':', alpha=0.7)
            
            # Plot segments as text annotations
            plt.subplot(2, 1, 2)
            plt.title("Lyric Segments Timeline")
            plt.xlabel("Time (seconds)")
            plt.yticks([])  # Hide y-axis
            
            # Set x-axis limits to match audio duration
            plt.xlim(0, self.duration)
            
            # Plot segment text at their start times
            for i, segment in enumerate(self.lyric_segments):
                plt.text(segment["start"], 0.5 + (i % 3) * 0.2, 
                         segment["text"], 
                         fontsize=8, 
                         ha='left', 
                         va='center',
                         bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                
                # Draw segment duration bars
                plt.hlines(0.5 + (i % 3) * 0.2, 
                          segment["start"], 
                          segment["end"], 
                          colors='green', 
                          linewidth=2)
            
            plt.tight_layout()
            plt.savefig(output_file)
            print(f"Visualization saved as '{output_file}'")
            return True
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False


def main():
    """Command-line interface for the lyrics extraction tool."""
    parser = argparse.ArgumentParser(description="Advanced Lyrics Extraction and Alignment")
    
    parser.add_argument("audio_file", help="Path to the vocal audio file")
    
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                        default="medium", help="Whisper model size to use")
    
    parser.add_argument("--language", help="Language code (e.g., 'en', 'es', 'fr') or None for auto-detect")
    
    parser.add_argument("--output-dir", default="lyrics_output",
                        help="Output directory for generated lyrics")
    
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                        help="Task: transcribe in original language or translate to English")
    
    parser.add_argument("--refine", action="store_true",
                        help="Refine alignments using audio analysis")
    
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA for GPU acceleration if available")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File {args.audio_file} not found.")
        return
    
    # Set device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # Create lyrics extractor
    extractor = LyricsExtractor(device=device, model_size=args.model)
    
    if not extractor.load_audio(args.audio_file):
        print("Failed to load audio file.")
        return
    
    # Load model and transcribe
    if extractor.load_model():
        print("\nTranscribing lyrics...")
        extractor.transcribe(language=args.language, task=args.task)
        
        # Refine alignments if requested
        if args.refine:
            print("\nRefining alignments...")
            extractor.refine_alignments()
        
        # Export in various formats
        print("\nExporting lyrics...")
        os.makedirs(args.output_dir, exist_ok=True)
        extractor.export_lyrics(output_dir=args.output_dir)
        
        # Generate visualization
        print("\nGenerating visualization...")
        extractor.visualize_alignments(
            output_file=os.path.join(args.output_dir, "lyrics_alignment.png")
        )
        
        print(f"\nComplete! All outputs saved to {args.output_dir}/")
    else:
        print("Failed to initialize the transcription model.")


if __name__ == "__main__":
    main()