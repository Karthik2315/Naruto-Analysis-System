import gradio as gr
import pandas as pd
import plotly.express as px
from theme_classifier import ThemeClassifier
from character_network import CharacterNetworkGenerator,NamedEntityRecognizer

def get_themes(themes_list, subtitles_path, save_path):
    themes_list = themes_list.split(',')
    themes_list = [theme.strip() for theme in themes_list if theme.lower().strip() != 'dialogue']
    theme_classifier = ThemeClassifier(themes_list)
    
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    theme_scores = output_df[themes_list].sum().reset_index()
    theme_scores.columns = ["theme", "score"]
    
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

def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_path)
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    
    return html

'''def main():
# theme classification 
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
# Character Network
    with gr.Blocks() as iface:
        gr.Markdown("## Character Network")

        with gr.Row():
            with gr.Column():
                network_html = gr.HTML()
            with gr.Column():
                subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                ner_path = gr.Textbox(label="NERs save path")
                get_character_network_button = gr.Button("Get Character Network Graph")

                get_character_network_button.click(
                    fn=get_character_network,
                    inputs=[subtitles_path, ner_path],
                    outputs=network_html
                )

    iface.launch() '''
    
def main():
    with gr.Blocks() as iface:
        gr.Markdown("## Naruto Theme Classification")
        
        # --- Row for Theme Classification ---
        with gr.Row():
            with gr.Column():
                theme_plot_output = gr.Plot(label="Theme Plot")
            with gr.Column():
                theme_list_input = gr.Textbox(label="Themes (comma separated)")
                themes_subtitles_path_input = gr.Textbox(label="Subtitles or Script Path")
                themes_save_path_input = gr.Textbox(label="Save Path")
                get_themes_button = gr.Button("Get Themes")

                get_themes_button.click(
                    fn=get_themes,
                    inputs=[theme_list_input, themes_subtitles_path_input, themes_save_path_input],
                    outputs=theme_plot_output
                )

        gr.Markdown("## Naruto Character Network")
        
        # --- Row for Character Network ---
        with gr.Row():
            with gr.Column():
                network_html = gr.HTML()
            with gr.Column():
                network_subtitles_path_input = gr.Textbox(label="Subtitles or Script Path")
                ner_path_input = gr.Textbox(label="NERs save path")
                get_character_network_button = gr.Button("Get Character Network Graph")

                get_character_network_button.click(
                    fn=get_character_network,
                    inputs=[network_subtitles_path_input, ner_path_input],
                    outputs=network_html
                )

    iface.launch()

if __name__ == "__main__":
    main()
