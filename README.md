# Music Analysis Toolkit

A comprehensive suite of professional-grade tools for music analysis, including chord recognition, track separation, guitar/bass tablature generation, and lyrics extraction.

## Features

### 1. Advanced Chord Analyzer
- Recognizes complex chords (7th, 9th, sus4, etc.)
- Detects chord progressions and suggests musical keys
- Visualizes chord changes over time
- Suggests scales for improvisation
- Supports multiple segmentation methods

### 2. Advanced Track Separator
- State-of-the-art source separation using:
  - Demucs
  - Open-Unmix
  - Spleeter
- Vocal isolation and enhancement
- Instrument-specific isolation (guitar, bass, etc.)
- Lyrics extraction using Whisper
- Stem visualization and analysis

### 3. Guitar/Bass Analyzer
- Note and chord detection
- String and fret position estimation
- Playing technique detection (bends, slides, hammer-ons)
- Tablature generation
- Supports alternate tunings
- Visualization of notes on fretboard

### 4. Lyrics Extractor
- High-quality transcription using Whisper
- Time-aligned lyrics with timestamps
- Word-level alignment for karaoke
- Multiple language support
- Confidence scoring for each segment

### 5. Music Master Analyzer (Integration)
- Orchestrates all components into a single pipeline
- Generates comprehensive analysis reports
- Produces human-readable summaries
- Manages output organization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mladjom/music.git
cd music
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Additional dependencies (optional but recommended):
```bash
pip install demucs openunmix spleeter openai-whisper
```

## Usage

### Individual Components

1. **Chord Analysis**:
```bash
python advanced_chord_analyzer.py audio_file.mp3 [--segmentation=adaptive|fixed|onset]
```

2. **Track Separation**:
```bash
python advanced_track_separator.py audio_file.mp3 [--method=demucs|openunmix|spleeter] [--stems=4|5|6]
```

3. **Guitar/Bass Analysis**:
```bash
python guitar_analyzer.py audio_file.mp3 [--instrument=guitar|bass] [--tuning=standard|drop_d|custom] [--detect-chords]
```

4. **Lyrics Extraction**:
```bash
python lyrics_extractor.py audio_file.mp3 [--model=medium|large] [--language=en|es|fr]
```

### Full Pipeline

For comprehensive analysis:
```bash
python music_master_analyzer.py audio_file.mp3 [--separation=demucs] [--no-guitar] [--analyze-bass] [--no-lyrics]
```

## Output Structure

All outputs are organized in the specified output directory (default: `music_analysis_output/`):
```
music_analysis_output/
├── separated_tracks/       # Separated audio stems
├── lyrics/                # Lyrics files in multiple formats
├── guitar_analysis/       # Guitar-specific analysis
├── bass_analysis/         # Bass-specific analysis
├── chord_analysis/        # Chord and music theory analysis
├── tablature/             # Generated guitar/bass tabs
├── analysis_report.json   # Combined analysis report
└── analysis_summary.txt   # Human-readable summary
```

## Requirements

- Python 3.7+
- Librosa
- NumPy
- SciPy
- Matplotlib
- Soundfile
- Torch (for GPU acceleration)
- (Optional) Demucs, Open-Unmix, Spleeter, Whisper

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Librosa for audio analysis
- Demucs, Open-Unmix, and Spleeter teams for separation models
- OpenAI for Whisper
- All open-source contributors to the dependencies