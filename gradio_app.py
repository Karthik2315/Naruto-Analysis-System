import gradio as gr
import pandas as pd
import plotly.express as px
from theme_classifier import ThemeClassifier

def get_themes(themes_list, subtitles_path, save_path):
    themes_list = themes_list.split(',')
    themes_list = [theme.strip() for theme in themes_list if theme.lower().strip() != 'dialogue']
    theme_classifier = ThemeClassifier(themes_list)
    
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    theme_scores = output_df[themes_list].sum().reset_index()
    theme_scores.columns = ["theme", "score"]
    
    # Create Plotly bar chart
    fig = px.bar(
        theme_scores,
        x="score",
        y="theme",
        orientation='h',
        title="Series Themes",
        labels={"score": "Score", "theme": "Theme"},
        height=350,
        width=750
    )
    return fig

def main():
    with gr.Blocks() as iface:
        gr.Markdown("## Naruto Theme Classification")

        with gr.Row():
            with gr.Column():
                plot_output = gr.Plot(label="Theme Plot")  
            with gr.Column():
                theme_list = gr.Textbox(label="Themes (comma separated)")
                subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                save_path = gr.Textbox(label="Save Path")
                get_themes_button = gr.Button("Get Themes")

                get_themes_button.click(
                    fn=get_themes,
                    inputs=[theme_list, subtitles_path, save_path],
                    outputs=plot_output
                )

    iface.launch(share=True)

if __name__ == "__main__":
    main()
