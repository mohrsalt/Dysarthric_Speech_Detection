import plotly.graph_objects as go

def create_confidence_bar_plot(data):
    # Filter only relevant entries
    filtered_data = [
        entry for entry in data
        if entry["confidence"] is not None and entry["prediction"] in ["dysarthric", "normal"]
    ]

    # Prepare the data for plotting
    x_vals = [(entry["start_time"] + entry["end_time"]) / 2 for entry in filtered_data]
    widths = [entry["end_time"] - entry["start_time"] for entry in filtered_data]
    y_vals = [entry["confidence"] for entry in filtered_data]
    colors = ['red' if entry["prediction"] == "dysarthric" else 'green' for entry in filtered_data]

    # Create the plotly figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        width=widths,
        marker_color=colors,
        name="Confidence",
        hovertext=[f"{entry['prediction']} ({entry['confidence']})" for entry in filtered_data],
    ))

    # Update layout for the figure
    fig.update_layout(
        title="Confidence of Dysarthric Predictions Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Confidence",
        bargap=0.1,
        showlegend=False,
        xaxis=dict(tickmode='linear')
    )

    return fig
