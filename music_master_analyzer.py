#!/usr/bin/env python3
# Professional Music Analysis Pipeline
# Integrates all components into a comprehensive analysis system

import os
import sys
import time
import json
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from advanced_track_separator import AdvancedTrackSeparator
from lyrics_extractor import LyricsExtractor
from guitar_analyzer import GuitarAnalyzer
from advanced_chord_analyzer import AdvancedChordAnalyzer

class MusicMasterAnalyzer:
    """
    Professional-grade music analysis pipeline that integrates:
    - Advanced track separation
    - Lyrics extraction and alignment
    - Guitar/bass note and chord detection
    - Tablature generation
    - Comprehensive music theory analysis
    
    This master class orchestrates the entire analysis process.
    """
    
    def __init__(self, output_dir="music_analysis_output"):
        """Initialize the master analyzer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each component
        self.stems_dir = os.path.join(output_dir, "separated_tracks")
        self.lyrics_dir = os.path.join(output_dir, "lyrics")
        self.guitar_dir = os.path.join(output_dir, "guitar_analysis")
        self.chord_dir = os.path.join(output_dir, "chord_analysis")
        self.tabs_dir = os.path.join(output_dir, "tablature")
        
        for directory in [self.stems_dir, self.lyrics_dir, self.guitar_dir, 
                         self.chord_dir, self.tabs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Component analyzers (initialized as needed)
        self.track_separator = None
        self.lyrics_extractor = None
        self.guitar_analyzer = None
        self.chord_analyzer = None
        
        # Result storage
        self.analysis_results = {}
        self.combined_report = {}
    
    def load_audio(self, file_path):
        """Load the main audio file for analysis."""
        try:
            # Use librosa for initial loading
            y, sr = librosa.load(file_path, sr=None)
            
            self.audio = y
            self.sr = sr
            self.duration = librosa.get_duration(y=y, sr=sr)
            self.file_path = file_path
            self.file_name = os.path.basename(file_path)
            
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            
            # Create summary in analysis results
            self.analysis_results['audio_info'] = {
                'file_path': file_path,
                'file_name': self.file_name,
                'duration': self.duration,
                'sample_rate': self.sr
            }
            
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def separate_tracks(self, method='demucs', extract_vocals=True):
        """
        Separate the audio into component tracks using the AdvancedTrackSeparator.
        
        Parameters:
        - method: Separation method ('demucs', 'openunmix', or 'spleeter')
        - extract_vocals: Whether to perform specialized vocal extraction
        """
        print("\n=== TRACK SEPARATION ===")
        start_time = time.time()
        
        # Initialize the track separator
        self.track_separator = AdvancedTrackSeparator()
        
        # Load the audio file
        if not self.track_separator.load_audio(self.file_path):
            print("Failed to load audio for track separation.")
            return False
        
        # Run the separation pipeline
        success = self.track_separator.run_full_separation(
            method=method,
            extract_lyrics=extract_vocals,
            isolate_instruments=True
        )
        
        if success:
            # Save the stems
            self.track_separator.save_all_stems(output_dir=self.stems_dir)
            
            # Generate visualization
            self.track_separator.visualize_separation(
                output_file=os.path.join(self.stems_dir, "track_separation.png")
            )
            
            # Analyze the stems
            stem_analysis = self.track_separator.analyze_stems()
            
            # Store results
            self.analysis_results['track_separation'] = {
                'method': method,
                'stems': list(self.track_separator.separated_stems.keys()),
                'isolated_instruments': list(self.track_separator.isolated_instruments.keys()),
                'stem_analysis': stem_analysis
            }
            
            elapsed_time = time.time() - start_time
            print(f"Track separation completed in {elapsed_time:.2f} seconds")
            return True
        else:
            print("Track separation failed.")
            return False
    
    def extract_lyrics(self, model_size="medium", refine_alignment=True):
        """
        Extract and analyze lyrics from the vocal track.
        
        Parameters:
        - model_size: Whisper model size for transcription
        - refine_alignment: Whether to refine timing of lyric alignments
        """
        if not self.track_separator or 'vocals' not in self.track_separator.separated_stems:
            print("No vocal stem available. Please run track separation first.")
            return False
        
        print("\n=== LYRICS EXTRACTION ===")
        start_time = time.time()
        
        # Save vocals to a temporary file
        vocals = self.track_separator.separated_stems['vocals']
        if len(vocals.shape) > 1 and vocals.shape[0] > 1:
            # Convert stereo to mono if needed
            vocals_mono = librosa.to_mono(vocals)
        else:
            vocals_mono = vocals
            
        temp_vocals_file = os.path.join(self.stems_dir, "vocals_for_lyrics.wav")
        librosa.output.write_wav(temp_vocals_file, vocals_mono, self.sr)
        
        # Initialize lyrics extractor
        self.lyrics_extractor = LyricsExtractor(model_size=model_size)
        
        # Load the vocal track
        if not self.lyrics_extractor.load_audio(temp_vocals_file):
            print("Failed to load vocals for lyrics extraction.")
            return False
        
        # Load model and transcribe
        if self.lyrics_extractor.load_model():
            # Transcribe vocals
            transcribe_success = self.lyrics_extractor.transcribe()
            
            if transcribe_success:
                # Refine alignments if requested
                if refine_alignment:
                    self.lyrics_extractor.refine_alignments()
                
                # Export lyrics in various formats
                self.lyrics_extractor.export_lyrics(output_dir=self.lyrics_dir)
                
                # Create visualization
                self.lyrics_extractor.visualize_alignments(
                    output_file=os.path.join(self.lyrics_dir, "lyrics_alignment.png")
                )
                
                # Generate karaoke-style lyrics
                karaoke_lyrics = self.lyrics_extractor.generate_karaoke_lyrics()
                
                # Store results
                self.analysis_results['lyrics'] = {
                    'text': self.lyrics_extractor.lyrics,
                    'language': self.lyrics_extractor.language,
                    'segment_count': len(self.lyrics_extractor.lyric_segments),
                    'karaoke_line_count': len(karaoke_lyrics)
                }
                
                elapsed_time = time.time() - start_time
                print(f"Lyrics extraction completed in {elapsed_time:.2f} seconds")
                return True
            else:
                print("Lyrics transcription failed.")
                return False
        else:
            print("Failed to initialize lyrics model.")
            return False
    
    def analyze_guitar(self, instrument_type='guitar', detect_chords=True):
        """
        Analyze guitar or bass parts to detect notes, chords, and generate tablature.
        
        Parameters:
        - instrument_type: 'guitar' or 'bass'
        - detect_chords: Whether to detect chords in addition to notes
        """
        # Check if we have the appropriate stem
        stem_key = instrument_type
        
        if not self.track_separator:
            print("No track separation available. Will use full mix for guitar analysis.")
            audio_file = self.file_path
        elif stem_key in self.track_separator.separated_stems:
            # Save the instrument stem to a file
            stem_audio = self.track_separator.separated_stems[stem_key]
            audio_file = os.path.join(self.stems_dir, f"{stem_key}_for_analysis.wav")
            
            if len(stem_audio.shape) > 1 and stem_audio.shape[0] > 1:
                # For stereo, use the first channel
                stem_audio_mono = stem_audio[0]
            else:
                stem_audio_mono = stem_audio
                
            librosa.output.write_wav(audio_file, stem_audio_mono, self.sr)
        elif f"enhanced_{stem_key}" in self.track_separator.isolated_instruments:
            # Use enhanced isolated instrument if available
            stem_audio = self.track_separator.isolated_instruments[f"enhanced_{stem_key}"]
            audio_file = os.path.join(self.stems_dir, f"enhanced_{stem_key}_for_analysis.wav")
            librosa.output.write_wav(audio_file, stem_audio, self.sr)
        else:
            print(f"No {instrument_type} stem available. Will use full mix for analysis.")
            audio_file = self.file_path
        
        print(f"\n=== {instrument_type.upper()} ANALYSIS ===")
        start_time = time.time()
        
        # Initialize guitar analyzer
        self.guitar_analyzer = GuitarAnalyzer(instrument=instrument_type)
        
        # Load the audio
        if not self.guitar_analyzer.load_audio(audio_file):
            print(f"Failed to load audio for {instrument_type} analysis.")
            return False
        
        # Detect notes
        self.guitar_analyzer.detect_notes()
        
        # Detect chords if requested
        if detect_chords:
            self.guitar_analyzer.detect_chords()
        
        # Generate visualization
        self.guitar_analyzer.visualize_notes(
            output_file=os.path.join(self.guitar_dir, f"{instrument_type}_notes.png")
        )
        
        # Generate tablature
        tab_lines = self.guitar_analyzer.generate_tablature(
            use_chords=detect_chords,
            optimize_positions=True,
            include_techniques=True
        )
        
        # Export results
        self.guitar_analyzer.export_tablature(
            filename=os.path.join(self.tabs_dir, f"{instrument_type}_tab.txt")
        )
        
        self.guitar_analyzer.export_to_json(
            filename=os.path.join(self.guitar_dir, f"{instrument_type}_analysis.json")
        )
        
        # Store results
        self.analysis_results[f'{instrument_type}_analysis'] = {
            'note_count': len(self.guitar_analyzer.note_events),
            'chord_count': len(self.guitar_analyzer.chord_events),
            'tab_lines': len(tab_lines)
        }
        
        elapsed_time = time.time() - start_time
        print(f"{instrument_type.capitalize()} analysis completed in {elapsed_time:.2f} seconds")
        return True
    
    def analyze_chords(self, segmentation='adaptive'):
        """
        Perform comprehensive chord and music theory analysis.
        
        Parameters:
        - segmentation: Method for chord segmentation ('adaptive', 'fixed', or 'onset')
        """
        print("\n=== CHORD ANALYSIS ===")
        start_time = time.time()
        
        # Initialize chord analyzer
        self.chord_analyzer = AdvancedChordAnalyzer()
        
        # Load the audio file
        if not self.chord_analyzer.load_audio(self.file_path):
            print("Failed to load audio for chord analysis.")
            return False
        
        # Extract features
        self.chord_analyzer.extract_features()
        
        # Detect chords
        chord_timeline = self.chord_analyzer.recognize_chords(segmentation=segmentation)
        
        if chord_timeline:
            # Analyze chord progression
            progression_analysis = self.chord_analyzer.analyze_progression()
            
            # Get scale suggestions
            scale_suggestions = self.chord_analyzer.suggest_scales()
            
            # Generate visualization
            self.chord_analyzer.visualize_chords(
                output_file=os.path.join(self.chord_dir, "chord_analysis.png")
            )
            
            # Export results
            self.chord_analyzer.export_to_json(
                filename=os.path.join(self.chord_dir, "chord_analysis.json")
            )
            
            # Store results
            self.analysis_results['chord_analysis'] = {
                'chord_count': len(chord_timeline),
                'progression_analysis': progression_analysis,
                'suggested_scales': scale_suggestions
            }
            
            elapsed_time = time.time() - start_time
            print(f"Chord analysis completed in {elapsed_time:.2f} seconds")
            return True
        else:
            print("No chords detected.")
            return False
    
    def generate_combined_report(self):
        """Generate a comprehensive analysis report combining all components."""
        if not self.analysis_results:
            print("No analysis results available.")
            return False
        
        # Basic song information
        self.combined_report = {
            'file_info': {
                'filename': self.file_name,
                'duration': self.duration,
                'sample_rate': self.sr
            },
            'analysis_components': list(self.analysis_results.keys())
        }
        
        # Add track separation summary
        if 'track_separation' in self.analysis_results:
            separation = self.analysis_results['track_separation']
            self.combined_report['track_separation'] = {
                'method': separation['method'],
                'stems': separation['stems'],
                'isolated_instruments': separation['isolated_instruments']
            }
        
        # Add lyrics summary
        if 'lyrics' in self.analysis_results:
            lyrics = self.analysis_results['lyrics']
            self.combined_report['lyrics'] = {
                'language': lyrics['language'],
                'segment_count': lyrics['segment_count'],
                'excerpt': lyrics['text'][:200] + "..." if len(lyrics['text']) > 200 else lyrics['text']
            }
        
        # Add chord analysis summary
        if 'chord_analysis' in self.analysis_results:
            chord_analysis = self.analysis_results['chord_analysis']
            
            # Get key information
            keys = []
            if 'progression_analysis' in chord_analysis and 'possible_keys' in chord_analysis['progression_analysis']:
                for key, score in chord_analysis['progression_analysis']['possible_keys']:
                    keys.append({'key': key, 'confidence': score})
            
            # Get scale suggestions
            scales = []
            if 'suggested_scales' in chord_analysis:
                for scale in chord_analysis['suggested_scales']:
                    scales.append({'name': scale['scale'], 'notes': scale['notes']})
            
            self.combined_report['music_theory'] = {
                'detected_chords': chord_analysis['chord_count'],
                'possible_keys': keys,
                'suggested_scales': scales
            }
        
        # Add guitar/bass analysis
        for instrument in ['guitar', 'bass']:
            key = f'{instrument}_analysis'
            if key in self.analysis_results:
                self.combined_report[f'{instrument}_analysis'] = {
                    'detected_notes': self.analysis_results[key]['note_count'],
                    'detected_chords': self.analysis_results[key]['chord_count'],
                    'tablature_generated': True
                }
        
        # Export the combined report
        report_file = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.combined_report, f, indent=2)
        
        print(f"Combined analysis report saved to {report_file}")
        
        # Also generate a text summary
        self._generate_text_summary()
        
        return True
    
    def _generate_text_summary(self):
        """Generate a human-readable text summary of the analysis."""
        summary_lines = []
        
        # Add header
        summary_lines.append("==================================")
        summary_lines.append("COMPREHENSIVE MUSIC ANALYSIS REPORT")
        summary_lines.append("==================================")
        summary_lines.append("")
        
        # Add file info
        file_info = self.combined_report['file_info']
        summary_lines.append(f"FILE: {file_info['filename']}")
        summary_lines.append(f"DURATION: {file_info['duration']:.2f} seconds ({int(file_info['duration']/60)}:{int(file_info['duration']%60):02d})")
        summary_lines.append("")
        
        # Add track separation info
        if 'track_separation' in self.combined_report:
            sep_info = self.combined_report['track_separation']
            summary_lines.append("TRACK SEPARATION:")
            summary_lines.append(f"Method: {sep_info['method']}")
            summary_lines.append(f"Separated stems: {', '.join(sep_info['stems'])}")
            if sep_info['isolated_instruments']:
                summary_lines.append(f"Isolated instruments: {', '.join(sep_info['isolated_instruments'])}")
            summary_lines.append("")
        
        # Add music theory info
        if 'music_theory' in self.combined_report:
            theory_info = self.combined_report['music_theory']
            summary_lines.append("MUSIC THEORY ANALYSIS:")
            summary_lines.append(f"Detected {theory_info['detected_chords']} chord changes")
            
            if 'possible_keys' in theory_info and theory_info['possible_keys']:
                summary_lines.append("\nPossible keys:")
                for key_info in theory_info['possible_keys'][:3]:  # Top 3 keys
                    summary_lines.append(f"  - {key_info['key']} (confidence: {key_info['confidence']:.2f})")
            
            if 'suggested_scales' in theory_info and theory_info['suggested_scales']:
                summary_lines.append("\nRecommended scales for improvisation:")
                for scale_info in theory_info['suggested_scales'][:3]:  # Top 3 scales
                    summary_lines.append(f"  - {scale_info['name']}: {', '.join(scale_info['notes'])}")
            
            summary_lines.append("")
        
        # Add instrument analysis info
        for instrument in ['guitar', 'bass']:
            key = f'{instrument}_analysis'
            if key in self.combined_report:
                inst_info = self.combined_report[key]
                summary_lines.append(f"{instrument.upper()} ANALYSIS:")
                summary_lines.append(f"Detected {inst_info['detected_notes']} notes")
                if inst_info['detected_chords'] > 0:
                    summary_lines.append(f"Detected {inst_info['detected_chords']} chords")
                if inst_info['tablature_generated']:
                    summary_lines.append(f"Tablature generated: {self.tabs_dir}/{instrument}_tab.txt")
                summary_lines.append("")
        
        # Add lyrics info
        if 'lyrics' in self.combined_report:
            lyrics_info = self.combined_report['lyrics']
            summary_lines.append("LYRICS ANALYSIS:")
            summary_lines.append(f"Language: {lyrics_info['language']}")
            summary_lines.append(f"Lyric segments: {lyrics_info['segment_count']}")
            summary_lines.append("\nLyrics excerpt:")
            summary_lines.append(lyrics_info['excerpt'])
            summary_lines.append("")
        
        # Add file locations
        summary_lines.append("OUTPUT LOCATIONS:")
        summary_lines.append(f"Separated tracks: {self.stems_dir}")
        summary_lines.append(f"Lyrics files: {self.lyrics_dir}")
        summary_lines.append(f"Chord analysis: {self.chord_dir}")
        summary_lines.append(f"Guitar/bass analysis: {self.guitar_dir}")
        summary_lines.append(f"Tablature: {self.tabs_dir}")
        summary_lines.append("")
        
        # Write to file
        summary_file = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_lines))
        
        print(f"Human-readable summary saved to {summary_file}")
    
    def run_full_analysis(self, separation_method='demucs', analyze_guitar=True, 
                         analyze_bass=False, extract_lyrics=True):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        - separation_method: Track separation method to use
        - analyze_guitar: Whether to perform guitar-specific analysis
        - analyze_bass: Whether to perform bass-specific analysis
        - extract_lyrics: Whether to extract lyrics
        """
        # Track total runtime
        total_start_time = time.time()
        
        print("\n========================================")
        print("STARTING COMPREHENSIVE MUSIC ANALYSIS")
        print("========================================")
        
        # Step 1: Track separation
        print("\nStep 1: Separating audio tracks...")
        self.separate_tracks(method=separation_method, extract_vocals=extract_lyrics)
        
        # Step 2: Chord analysis
        print("\nStep 2: Analyzing chords and music theory...")
        self.analyze_chords()
        
        # Step 3: Instrument-specific analysis
        if analyze_guitar:
            print("\nStep 3a: Analyzing guitar parts...")
            self.analyze_guitar(instrument_type='guitar')
        
        if analyze_bass:
            print("\nStep 3b: Analyzing bass parts...")
            self.analyze_guitar(instrument_type='bass')
        
        # Step 4: Lyrics extraction (if vocals available)
        if extract_lyrics:
            print("\nStep 4: Extracting and aligning lyrics...")
            self.extract_lyrics()
        
        # Step 5: Generate combined report
        print("\nStep 5: Generating comprehensive report...")
        self.generate_combined_report()
        
        # Done!
        total_elapsed = time.time() - total_start_time
        print("\n========================================")
        print(f"ANALYSIS COMPLETE in {total_elapsed:.2f} seconds")
        print(f"All results saved to {self.output_dir}/")
        print("========================================")
        
        return True


def main():
    """Command-line interface for the master music analyzer."""
    parser = argparse.ArgumentParser(description="Professional Music Analysis Pipeline")
    
    parser.add_argument("audio_file", help="Path to the audio file to analyze")
    
    parser.add_argument("--output-dir", default="music_analysis_output",
                       help="Output directory for all analysis files")
    
    parser.add_argument("--separation", choices=["demucs", "openunmix", "spleeter"], 
                       default="demucs", help="Track separation method to use")
    
    parser.add_argument("--no-guitar", action="store_true",
                       help="Skip guitar analysis")
    
    parser.add_argument("--analyze-bass", action="store_true",
                       help="Include bass analysis")
    
    parser.add_argument("--no-lyrics", action="store_true",
                       help="Skip lyrics extraction")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File {args.audio_file} not found.")
        return
    
    # Create master analyzer
    analyzer = MusicMasterAnalyzer(output_dir=args.output_dir)
    
    # Load audio
    if not analyzer.load_audio(args.audio_file):
        print("Failed to load audio file.")
        return
    
    # Run full analysis with selected options
    analyzer.run_full_analysis(
        separation_method=args.separation,
        analyze_guitar=not args.no_guitar,
        analyze_bass=args.analyze_bass,
        extract_lyrics=not args.no_lyrics
    )


if __name__ == "__main__":
    main()