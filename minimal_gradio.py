import logging
import traceback
import gradio as gr

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def minimal_interface(text):
    return f"You entered: {text}"

def create_minimal_demo():
    demo = gr.Interface(
        fn=minimal_interface,
        inputs="text",
        outputs="text",
        title="Minimal XTTS Demo",
    )
    return demo

if __name__ == "__main__":
    try:
        demo = create_minimal_demo()
        demo.launch(server_name="0.0.0.0", server_port=8080, share=False, debug=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        exit(1)