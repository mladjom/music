#!/usr/bin/env python3
# Advanced Track Separator with Deep Learning
# Professional-grade music source separation tool

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import subprocess
import time
import json
import argparse
from pathlib import Path

class AdvancedTrackSeparator:
    """
    Professional-grade music source separation tool that supports multiple
    deep learning models and advanced processing techniques.
    
    Features:
    - Deep learning-based separation (Demucs, Open-Unmix, Spleeter)
    - Instrument-specific isolation
    - Vocal isolation and enhancement
    - Multi-stage processing pipeline
    - Advanced spectral processing
    """
    
    def __init__(self, device=None, model_dir=None):
        """Initialize the separator with processing parameters."""
        # Set up device (CPU or GPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model storage directory
        self.model_dir = model_dir or os.path.join(os.path.expanduser('~'), '.music_analyzer', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize state
        self.audio = None
        self.sr = None
        self.duration = None
        self.separated_stems = {}
        self.isolated_instruments = {}
        self.lyrics = None
        self.available_models = {
            'demucs': self._check_demucs_available(),
            'openunmix': self._check_openunmix_available(),
            'spleeter': self._check_spleeter_available()
        }
        
        # Print available models
        print("Available separation models:")
        for model, available in self.available_models.items():
            status = "✓" if available else "✗"
            print(f"  {model}: {status}")
    
    def _check_demucs_available(self):
        """Check if Demucs is available."""
        try:
            import demucs
            return True
        except ImportError:
            return False
    
    def _check_openunmix_available(self):
        """Check if Open-Unmix is available."""
        try:
            import openunmix
            return True
        except ImportError:
            return False
    
    def _check_spleeter_available(self):
        """Check if Spleeter is available."""
        try:
            # Try to run spleeter with no args to see if it's installed
            process = subprocess.run(
                ["spleeter", "--help"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5
            )
            return process.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def load_audio(self, file_path):
        """Load an audio file for separation."""
        try:
            # Use librosa for initial loading (handles more formats)
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Convert to mono if needed for processing
            if len(y.shape) > 1:
                print(f"Loaded stereo audio with {y.shape[0]} channels")
                # Keep original for later, but create mono version for analysis
                self.audio_stereo = y
                self.audio = librosa.to_mono(y)
            else:
                print(f"Loaded mono audio")
                self.audio = y
                # Create a "fake" stereo version for models that require it
                self.audio_stereo = np.stack([y, y])
            
            self.sr = sr
            self.duration = librosa.get_duration(y=self.audio, sr=sr)
            self.file_path = file_path
            
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            print(f"Sample rate: {self.sr} Hz")
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def separate_with_demucs(self, model_name='htdemucs', num_stems=4):
        """
        Separate tracks using Demucs, a state-of-the-art music source separation model.
        
        Parameters:
        - model_name: The Demucs model variant to use
          Options: 'htdemucs' (default), 'htdemucs_ft', 'mdx_extra', etc.
        - num_stems: Number of stems to extract (4 or 6)
          4 stems: drums, bass, vocals, other
          6 stems: drums, bass, vocals, piano, guitar, other
        """
        if not self.available_models['demucs']:
            print("Demucs is not available. Please install it with:")
            print("pip install demucs")
            return False
        
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            print(f"Loading Demucs model: {model_name}")
            model = get_model(model_name)
            model.to(self.device)
            
            # Convert audio to tensor format expected by Demucs
            audio_tensor = torch.tensor(self.audio_stereo, dtype=torch.float32).to(self.device)
            
            # Run separation
            print("Running Demucs separation...")
            with torch.no_grad():
                sources = apply_model(model, audio_tensor.unsqueeze(0))[0]
            
            # Process and store the separated stems
            stem_names = model.sources
            print(f"Extracted stems: {stem_names}")
            
            # Store separated sources
            for i, name in enumerate(stem_names):
                source_audio = sources[i].cpu().numpy()
                self.separated_stems[name] = source_audio
            
            print(f"Successfully separated into {len(stem_names)} stems using Demucs")
            return True
        except Exception as e:
            print(f"Error during Demucs separation: {e}")
            return False
    
    def separate_with_openunmix(self):
        """Separate tracks using Open-Unmix, an open-source music separation model."""
        if not self.available_models['openunmix']:
            print("Open-Unmix is not available. Please install it with:")
            print("pip install openunmix")
            return False
        
        try:
            import openunmix
            
            # Convert audio to tensor
            audio_tensor = torch.tensor(self.audio_stereo, dtype=torch.float32)
            
            # Open-Unmix expects input as (batch, channels, samples)
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            print("Running OpenUnmix separation...")
            
            # Run the separation models
            vocals = openunmix.separate.predict.separate(audio_tensor, rate=self.sr, model_name="umxhq")[0]
            drums = openunmix.separate.predict.separate(audio_tensor, rate=self.sr, model_name="umxhq-drums")[0]
            bass = openunmix.separate.predict.separate(audio_tensor, rate=self.sr, model_name="umxhq-bass")[0]
            
            # Create 'other' stem by subtracting
            other = audio_tensor[0] - vocals - drums - bass
            
            # Store results
            self.separated_stems["vocals"] = vocals.cpu().numpy()
            self.separated_stems["drums"] = drums.cpu().numpy()
            self.separated_stems["bass"] = bass.cpu().numpy()
            self.separated_stems["other"] = other.cpu().numpy()
            
            print("Successfully separated into 4 stems using Open-Unmix")
            return True
        except Exception as e:
            print(f"Error during Open-Unmix separation: {e}")
            return False
    
    def separate_with_spleeter(self, stems=5):
        """
        Separate tracks using Spleeter.
        
        Parameters:
        - stems: Number of stems (2, 4, or 5)
            - 2: vocals and accompaniment
            - 4: vocals, drums, bass, and other
            - 5: vocals, drums, bass, piano, and other
        """
        if not self.available_models['spleeter']:
            print("Spleeter is not available. Please install it with:")
            print("pip install spleeter")
            return False
        
        try:
            # Create a temporary file for Spleeter
            temp_audio_file = "temp_audio_for_spleeter.wav"
            output_dir = "spleeter_output"
            
            # Save audio to a temporary file
            sf.write(temp_audio_file, self.audio_stereo.T, self.sr)
            
            try:
                # Ensure stems is a valid option
                if stems not in [2, 4, 5]:
                    print("Invalid stems value. Must be 2, 4, or 5.")
                    return False
                
                # Call Spleeter via subprocess
                print(f"Running Spleeter to separate into {stems} stems...")
                model = f"spleeter:stems-{stems}"
                cmd = [
                    "spleeter", "separate", 
                    "-o", output_dir, 
                    "-p", model,
                    temp_audio_file
                ]
                
                start_time = time.time()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"Spleeter failed with error: {stderr.decode()}")
                    return False
                
                print(f"Spleeter separation completed in {time.time() - start_time:.2f} seconds.")
                
                # Load the separated tracks
                base_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(temp_audio_file))[0])
                
                if stems == 2:
                    components = ['vocals', 'accompaniment']
                elif stems == 4:
                    components = ['vocals', 'drums', 'bass', 'other']
                elif stems == 5:
                    components = ['vocals', 'drums', 'bass', 'piano', 'other']
                
                for component in components:
                    file_path = os.path.join(base_dir, f"{component}.wav")
                    if os.path.exists(file_path):
                        audio, _ = librosa.load(file_path, sr=self.sr, mono=False)
                        self.separated_stems[component] = audio
                        print(f"Loaded separated {component} track.")
                
                return True
                
            except Exception as e:
                print(f"Error using Spleeter: {e}")
                return False
            finally:
                # Clean up temporary files
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
        except Exception as e:
            print(f"Error during Spleeter separation: {e}")
            return False
    
    def isolate_vocal_with_spectral_masking(self):
        """
        Enhanced vocal isolation using spectral masking techniques.
        This can be used to further refine vocals from other separation methods.
        """
        if "vocals" not in self.separated_stems:
            print("No vocals stem found. Run a separation method first.")
            return False
        
        try:
            # Get the vocals from previous separation
            vocals = self.separated_stems["vocals"]
            
            # Convert to mono if stereo
            if len(vocals.shape) > 1 and vocals.shape[0] > 1:
                vocals_mono = librosa.to_mono(vocals)
            else:
                vocals_mono = vocals
            
            # Compute STFT
            stft = librosa.stft(vocals_mono)
            magnitude, phase = librosa.magphase(stft)
            
            # Apply soft mask to enhance vocals (reduce background noise)
            # This focuses on frequency ranges typical for human voice
            freq_bins = librosa.fft_frequencies(sr=self.sr)
            
            # Create a mask that emphasizes vocal frequencies (typically 200-8000 Hz)
            vocal_mask = np.ones_like(magnitude)
            
            # Reduce frequencies below typical vocal range
            vocal_mask[freq_bins < 200, :] *= 0.5
            
            # Reduce frequencies above typical vocal range
            vocal_mask[freq_bins > 8000, :] *= 0.3
            
            # Apply the mask
            enhanced_magnitude = magnitude * vocal_mask
            
            # Reconstruct the signal
            enhanced_stft = enhanced_magnitude * phase
            enhanced_vocals = librosa.istft(enhanced_stft)
            
            # Store the enhanced vocals
            self.isolated_instruments["enhanced_vocals"] = enhanced_vocals
            
            print("Vocal isolation and enhancement completed")
            return True
        except Exception as e:
            print(f"Error during vocal isolation: {e}")
            return False
    
    def isolate_guitar(self, refinement_level=2):
        """
        Advanced guitar isolation using spectral characteristics.
        Works best when 'other' stem is available from prior separation.
        
        Parameters:
        - refinement_level: 1-3, with 3 being the most aggressive isolation
        """
        # Check if we have the necessary stems
        if not any(stem in self.separated_stems for stem in ["other", "guitar"]):
            print("No suitable stem found for guitar isolation. Run a separation method first.")
            return False
        
        try:
            # Use guitar stem if available, otherwise use 'other'
            if "guitar" in self.separated_stems:
                source = self.separated_stems["guitar"]
            else:
                source = self.separated_stems["other"]
            
            # Convert to mono if needed
            if len(source.shape) > 1 and source.shape[0] > 1:
                source_mono = librosa.to_mono(source)
            else:
                source_mono = source
            
            # Compute STFT
            stft = librosa.stft(source_mono)
            magnitude, phase = librosa.magphase(stft)
            
            # Create a mask that emphasizes guitar frequencies
            freq_bins = librosa.fft_frequencies(sr=self.sr)
            guitar_mask = np.ones_like(magnitude)
            
            # Typical guitar frequency ranges
            # Low E: ~82 Hz, High E: ~330 Hz (fundamentals)
            # With harmonics extending to 5 kHz or more
            
            # Different refinement levels offer different isolation strategies
            if refinement_level == 1:
                # Gentle refinement - wide frequency band
                guitar_mask[freq_bins < 80, :] *= 0.5
                guitar_mask[freq_bins > 5000, :] *= 0.5
                
            elif refinement_level == 2:
                # Medium refinement - more focused on guitar range
                guitar_mask[freq_bins < 80, :] *= 0.3
                guitar_mask[freq_bins > 5000, :] *= 0.3
                # Emphasize fundamental guitar frequencies
                guitar_boost = (freq_bins >= 80) & (freq_bins <= 1200)
                guitar_mask[guitar_boost, :] *= 1.5
                
            elif refinement_level == 3:
                # Aggressive refinement - fine-tuned for guitar
                guitar_mask[freq_bins < 80, :] *= 0.1
                guitar_mask[freq_bins > 5000, :] *= 0.1
                # Emphasize fundamental guitar frequencies
                guitar_boost = (freq_bins >= 80) & (freq_bins <= 1200)
                guitar_mask[guitar_boost, :] *= 2.0
                # Further emphasize typical guitar harmony frequencies
                guitar_harmony = (freq_bins > 1200) & (freq_bins <= 3500)
                guitar_mask[guitar_harmony, :] *= 1.5
            
            # Apply the mask
            enhanced_magnitude = magnitude * guitar_mask
            
            # Reconstruct the signal
            enhanced_stft = enhanced_magnitude * phase
            enhanced_guitar = librosa.istft(enhanced_stft)
            
            # Store the enhanced guitar
            self.isolated_instruments["enhanced_guitar"] = enhanced_guitar
            
            print(f"Guitar isolation completed (refinement level: {refinement_level})")
            return True
        except Exception as e:
            print(f"Error during guitar isolation: {e}")
            return False
    
    def isolate_bass(self, refinement_level=2):
        """
        Advanced bass isolation using spectral characteristics.
        Works best when 'bass' stem is available from prior separation.
        
        Parameters:
        - refinement_level: 1-3, with 3 being the most aggressive isolation
        """
        # Check if we have the necessary stems
        if "bass" not in self.separated_stems:
            print("No bass stem found. Run a separation method first.")
            return False
        
        try:
            # Get the bass stem
            source = self.separated_stems["bass"]
            
            # Convert to mono if needed
            if len(source.shape) > 1 and source.shape[0] > 1:
                source_mono = librosa.to_mono(source)
            else:
                source_mono = source
            
            # Compute STFT
            stft = librosa.stft(source_mono)
            magnitude, phase = librosa.magphase(stft)
            
            # Create a mask that emphasizes bass frequencies
            freq_bins = librosa.fft_frequencies(sr=self.sr)
            bass_mask = np.ones_like(magnitude)
            
            # Different refinement levels offer different isolation strategies
            if refinement_level == 1:
                # Gentle refinement - include wider frequency range
                bass_mask[freq_bins > 500, :] *= 0.5
                
            elif refinement_level == 2:
                # Medium refinement - more focused on bass range
                bass_mask[freq_bins > 350, :] *= 0.3
                # Emphasize fundamental bass frequencies (40-200 Hz)
                bass_boost = (freq_bins >= 40) & (freq_bins <= 200)
                bass_mask[bass_boost, :] *= 1.5
                
            elif refinement_level == 3:
                # Aggressive refinement - fine-tuned for bass
                bass_mask[freq_bins > 250, :] *= 0.1
                # Emphasize fundamental bass frequencies (40-200 Hz)
                bass_boost = (freq_bins >= 40) & (freq_bins <= 200)
                bass_mask[bass_boost, :] *= 2.0
            
            # Apply the mask
            enhanced_magnitude = magnitude * bass_mask
            
            # Reconstruct the signal
            enhanced_stft = enhanced_magnitude * phase
            enhanced_bass = librosa.istft(enhanced_stft)
            
            # Store the enhanced bass
            self.isolated_instruments["enhanced_bass"] = enhanced_bass
            
            print(f"Bass isolation completed (refinement level: {refinement_level})")
            return True
        except Exception as e:
            print(f"Error during bass isolation: {e}")
            return False
    
    def extract_lyrics(self, use_whisper=True, language=None):
        """
        Extract lyrics from the vocal stem using speech recognition.
        
        Parameters:
        - use_whisper: Whether to use OpenAI's Whisper model (better for singing)
        - language: Target language code (e.g., 'en', 'es') or None for auto-detect
        """
        if "vocals" not in self.separated_stems and "enhanced_vocals" not in self.isolated_instruments:
            print("No vocal stem found. Run a separation and vocal isolation first.")
            return False
        
        try:
            # Use enhanced vocals if available, otherwise use regular vocals
            if "enhanced_vocals" in self.isolated_instruments:
                vocals = self.isolated_instruments["enhanced_vocals"]
            else:
                # If vocal stem is stereo, convert to mono
                vocals_stem = self.separated_stems["vocals"]
                if len(vocals_stem.shape) > 1 and vocals_stem.shape[0] > 1:
                    vocals = librosa.to_mono(vocals_stem)
                else:
                    vocals = vocals_stem
            
            # Save vocals to a temporary file for speech recognition
            temp_vocals_file = "temp_vocals_for_recognition.wav"
            sf.write(temp_vocals_file, vocals, self.sr)
            
            try:
                if use_whisper:
                    try:
                        import whisper
                    except ImportError:
                        print("Whisper not found. Please install it with:")
                        print("pip install openai-whisper")
                        return False
                    
                    print("Loading Whisper model...")
                    # Load a medium model for better accuracy with music
                    model = whisper.load_model("medium")
                    
                    print("Transcribing vocals...")
                    result = model.transcribe(
                        temp_vocals_file,
                        language=language,
                        task="transcribe"
                    )
                    
                    # Extract and store the lyrics
                    self.lyrics = result["text"]
                    
                    # Store timestamp segments for potential alignment
                    self.lyric_segments = result["segments"]
                    
                    print("Lyrics extraction completed")
                    print(f"Found {len(self.lyric_segments)} lyric segments")
                    return True
                else:
                    # Fallback to SpeechRecognition library if whisper is not used
                    try:
                        import speech_recognition as sr
                    except ImportError:
                        print("SpeechRecognition not found. Please install it with:")
                        print("pip install SpeechRecognition")
                        return False
                    
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(temp_vocals_file) as source:
                        audio_data = recognizer.record(source)
                        print("Recognizing speech...")
                        self.lyrics = recognizer.recognize_google(audio_data, language=language)
                        print("Lyrics extraction completed")
                        return True
            finally:
                # Clean up temporary file
                if os.path.exists(temp_vocals_file):
                    os.remove(temp_vocals_file)
                    
        except Exception as e:
            print(f"Error during lyrics extraction: {e}")
            return False
    
    def run_full_separation(self, method='demucs', extract_lyrics=True, isolate_instruments=True):
        """
        Run a complete separation workflow with all enhancements.
        
        Parameters:
        - method: The main separation method to use ('demucs', 'openunmix', 'spleeter')
        - extract_lyrics: Whether to extract lyrics from vocals
        - isolate_instruments: Whether to perform additional instrument isolation
        """
        if self.audio is None:
            print("No audio loaded. Please load an audio file first.")
            return False
        
        start_time = time.time()
        print(f"\nRunning full separation pipeline using {method}...")
        
        # Step 1: Initial stem separation
        if method == 'demucs':
            # Use 6-stem model for better instrument separation
            success = self.separate_with_demucs(model_name='htdemucs_6s', num_stems=6)
        elif method == 'openunmix':
            success = self.separate_with_openunmix()
        elif method == 'spleeter':
            # Use 5-stem model for better instrument coverage
            success = self.separate_with_spleeter(stems=5)
        else:
            print(f"Unknown separation method: {method}")
            return False
        
        if not success:
            print("Initial separation failed.")
            return False
        
        # Step 2: Enhance and isolate vocals
        print("\nEnhancing vocals...")
        self.isolate_vocal_with_spectral_masking()
        
        # Step 3: Extract lyrics if requested
        if extract_lyrics:
            print("\nExtracting lyrics...")
            self.extract_lyrics()
        
        # Step 4: Isolate additional instruments if requested
        if isolate_instruments:
            print("\nIsolating guitar...")
            self.isolate_guitar(refinement_level=2)
            
            print("\nEnhancing bass...")
            self.isolate_bass(refinement_level=2)
        
        elapsed_time = time.time() - start_time
        print(f"\nFull separation pipeline completed in {elapsed_time:.2f} seconds")
        
        # Summarize results
        all_stems = list(self.separated_stems.keys())
        all_instruments = list(self.isolated_instruments.keys())
        
        print(f"\nSeparated stems: {', '.join(all_stems)}")
        print(f"Isolated instruments: {', '.join(all_instruments)}")
        
        if self.lyrics:
            print(f"\nLyrics extracted: {len(self.lyrics)} characters")
        
        return True
    
    def save_all_stems(self, output_dir='separated_stems'):
        """Save all separated stems to individual audio files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save regular stems
        for name, audio in self.separated_stems.items():
            output_file = os.path.join(output_dir, f"{name}.wav")
            # Handle both mono and stereo stems
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                sf.write(output_file, audio.T, self.sr)  # Transpose for soundfile's expected format
            else:
                sf.write(output_file, audio, self.sr)
            print(f"Saved {name} stem to {output_file}")
        
        # Save isolated instruments
        for name, audio in self.isolated_instruments.items():
            output_file = os.path.join(output_dir, f"{name}.wav")
            sf.write(output_file, audio, self.sr)
            print(f"Saved {name} to {output_file}")
        
        # Save lyrics if available
        if self.lyrics:
            lyrics_file = os.path.join(output_dir, "lyrics.txt")
            with open(lyrics_file, 'w', encoding='utf-8') as f:
                f.write(self.lyrics)
            print(f"Saved lyrics to {lyrics_file}")
        
        return True
    
    def visualize_separation(self, output_file='track_separation.png'):
        """Visualize the separated stems and isolated instruments."""
        # Count how many waveforms to display
        num_waveforms = 1 + len(self.separated_stems) + len(self.isolated_instruments)
        
        # Create figure with appropriate size
        fig, axs = plt.subplots(num_waveforms, 1, figsize=(12, 2 * num_waveforms))
        
        # Plot original waveform
        axs[0].set_title('Original Audio')
        librosa.display.waveshow(self.audio, sr=self.sr, ax=axs[0])
        
        # Plot each stem
        for i, (name, audio) in enumerate(self.separated_stems.items(), 1):
            # Handle both mono and stereo stems
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                # For stereo, plot just the first channel for simplicity
                plot_audio = audio[0]
            else:
                plot_audio = audio
                
            axs[i].set_title(f'Stem: {name}')
            librosa.display.waveshow(plot_audio, sr=self.sr, ax=axs[i])
        
        # Plot each isolated instrument
        offset = 1 + len(self.separated_stems)
        for i, (name, audio) in enumerate(self.isolated_instruments.items(), 0):
            axs[offset + i].set_title(f'Isolated: {name}')
            librosa.display.waveshow(audio, sr=self.sr, ax=axs[offset + i])
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Visualization saved as '{output_file}'")
        
        return fig
    
    def analyze_stems(self):
        """Analyze characteristics of each separated stem."""
        analysis = {}
        
        # Analyze regular stems
        for name, audio in self.separated_stems.items():
            # Convert stereo to mono for analysis if needed
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                analysis_audio = librosa.to_mono(audio)
            else:
                analysis_audio = audio
                
            # Calculate RMS energy
            rms = librosa.feature.rms(y=analysis_audio)[0]
            
            # Calculate spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=analysis_audio, sr=self.sr)[0]
            
            # Calculate spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=analysis_audio, sr=self.sr)[0]
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=analysis_audio, sr=self.sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
            
            # Calculate tempo
            tempo, _ = librosa.beat.beat_track(y=analysis_audio, sr=self.sr)
            
            # Store analysis
            analysis[name] = {
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'centroid_mean': float(np.mean(centroid)),
                'centroid_std': float(np.std(centroid)),
                'bandwidth_mean': float(np.mean(bandwidth)),
                'bandwidth_std': float(np.std(bandwidth)),
                'onset_count': int(len(onsets)),
                'tempo': float(tempo.item() if hasattr(tempo, 'item') else tempo)
            }
        
        # Analyze isolated instruments
        for name, audio in self.isolated_instruments.items():
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Calculate spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            
            # Calculate spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
            
            # Calculate tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
            
            # Store analysis
            analysis[name] = {
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'centroid_mean': float(np.mean(centroid)),
                'centroid_std': float(np.std(centroid)),
                'bandwidth_mean': float(np.mean(bandwidth)),
                'bandwidth_std': float(np.std(bandwidth)),
                'onset_count': int(len(onsets)),
                'tempo': float(tempo.item() if hasattr(tempo, 'item') else tempo)
            }
        
        # Print analysis
        print("\nStem Analysis:")
        for name, metrics in analysis.items():
            print(f"\n{name.upper()}:")
            print(f"  Energy (RMS): {metrics['rms_mean']:.4f} ± {metrics['rms_std']:.4f}")
            print(f"  Brightness (Centroid): {metrics['centroid_mean']:.1f} Hz ± {metrics['centroid_std']:.1f} Hz")
            print(f"  Bandwidth: {metrics['bandwidth_mean']:.1f} Hz ± {metrics['bandwidth_std']:.1f} Hz")
            print(f"  Onset Count: {metrics['onset_count']}")
            print(f"  Estimated Tempo: {metrics['tempo']:.1f} BPM")
        
        # Save analysis to JSON
        with open('stem_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("\nAnalysis saved to 'stem_analysis.json'")
        return analysis
    
    def export_to_json(self, filename="separation_results.json"):
        """Export separation metadata to JSON file."""
        export_data = {
            "audio_file": self.file_path,
            "duration": self.duration,
            "sample_rate": self.sr,
            "separation_results": {
                "stems": list(self.separated_stems.keys()),
                "isolated_instruments": list(self.isolated_instruments.keys()),
                "lyrics_extracted": self.lyrics is not None
            }
        }
        
        # Add lyrics if available
        if self.lyrics:
            export_data["lyrics"] = self.lyrics
            if hasattr(self, 'lyric_segments'):
                export_data["lyric_segments"] = [
                    {
                        "start": segment["start"], 
                        "end": segment["end"],
                        "text": segment["text"]
                    } for segment in self.lyric_segments
                ]
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Separation results exported to {filename}")
        return filename


def install_dependencies():
    """Helper function to install required dependencies."""
    dependencies = [
        ("torch", "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"),
        ("librosa", "pip install librosa"),
        ("soundfile", "pip install soundfile"),
        ("matplotlib", "pip install matplotlib"),
        ("demucs", "pip install demucs"),
        ("openunmix", "pip install openunmix"),
        ("spleeter", "pip install spleeter"),
        ("whisper", "pip install openai-whisper")
    ]
    
    print("Checking for required dependencies...")
    missing = []
    
    for package, install_cmd in dependencies:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing.append((package, install_cmd))
    
    if missing:
        print("\nMissing dependencies found. Install them with the following commands:")
        for package, install_cmd in missing:
            print(f"  {install_cmd}")
        return False
    else:
        print("All dependencies are installed!")
        return True


def main():
    """Command-line interface for the advanced track separator."""
    parser = argparse.ArgumentParser(description="Advanced Track Separator with Deep Learning")
    
    parser.add_argument("audio_file", help="Path to the audio file to process")
    
    parser.add_argument("--method", choices=["demucs", "openunmix", "spleeter"], default="demucs",
                       help="Separation method to use")
    
    parser.add_argument("--stems", type=int, default=6, choices=[2, 4, 5, 6],
                       help="Number of stems for Spleeter or Demucs (2, 4, 5, or 6)")
    
    parser.add_argument("--no-lyrics", action="store_true",
                       help="Skip lyrics extraction")
    
    parser.add_argument("--no-instrument-isolation", action="store_true",
                       help="Skip additional instrument isolation")
    
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies and exit")
    
    parser.add_argument("--output-dir", default="separated_stems",
                       help="Output directory for separated stems")
    
    parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA for GPU acceleration if available")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        install_dependencies()
        return
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File {args.audio_file} not found.")
        return
    
    # Set device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # Create separator and load audio
    separator = AdvancedTrackSeparator(device=device)
    
    if not separator.load_audio(args.audio_file):
        print("Failed to load audio file.")
        return
    
    # Run full separation pipeline
    extract_lyrics = not args.no_lyrics
    isolate_instruments = not args.no_instrument_isolation
    
    separator.run_full_separation(
        method=args.method,
        extract_lyrics=extract_lyrics,
        isolate_instruments=isolate_instruments
    )
    
    # Save outputs
    separator.save_all_stems(output_dir=args.output_dir)
    
    # Visualize and analyze
    separator.visualize_separation()
    separator.analyze_stems()
    
    # Export metadata
    separator.export_to_json()
    
    print(f"\nComplete! All outputs saved to {args.output_dir}/")


