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
        'Abnormal Speech': 'red',
        'Normal Speech': 'blue',
        'Non-speech': 'black'
    }

    # Initialize plot
    fig = go.Figure()

    # Track which labels have already been added to the legend
    added_labels = set()
    
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

        # Show legend only once per label
        show_legend = label not in added_labels
        if show_legend:
            added_labels.add(label)

        # Plot segment
        fig.add_trace(go.Scatter(
            x=time[start_sample:end_sample],
            y=y[start_sample:end_sample],
            mode='lines',
            line=dict(color=color),
            name=label,
            showlegend=show_legend
        ))

    fig.update_layout(
        title='Audio Waveform with Segment-Based Predictions',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=500,
        width=1400,  # Increased width for a wider plot
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
