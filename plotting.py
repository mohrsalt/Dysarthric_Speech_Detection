import numpy as np
import librosa
import plotly.graph_objs as go

def plot_colored_waveform(audio_path, prediction_segments):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Normalize the waveform
    y = y / np.max(np.abs(y))
    
    # Generate time axis
    duration = len(y) / sr
    time = np.linspace(0, duration, num=len(y))

    # Mapping labels to colors
    label_to_color = {
        'Dysarthric': 'red',
        'Control': 'blue',
        'non-speech': 'black'
    }

    # Initialize plot
    fig = go.Figure()
    
    # Process each segment
    for segment in prediction_segments:
        start = segment['start_time']
        end = segment['end_time']
        label = str(segment['prediction'])  # In case it's np.str_
        color = label_to_color.get(label, 'gray')  # fallback color

        # Get sample range for this segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Clip indices to avoid overflow
        start_sample = max(0, min(start_sample, len(y)))
        end_sample = max(0, min(end_sample, len(y)))

        # Plot segment
        fig.add_trace(go.Scatter(
            x=time[start_sample:end_sample],
            y=y[start_sample:end_sample],
            mode='lines',
            line=dict(color=color),
            name=label,
            showlegend=False  # Hide legend for now
        ))

    fig.update_layout(
        title='Audio Waveform with Segment-Based Predictions',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=300
    )
    
    return fig
