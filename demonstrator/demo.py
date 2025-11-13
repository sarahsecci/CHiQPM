import gradio as gr
import torch
import torchvision.transforms as transforms
from pathlib import Path
from evaluation.load_model import get_args_for_loading_model, load_model
from PIL import Image
from dataset_classes.cub200 import load_cub_class_mapping
from get_data import DeepLabBirdCrop, get_data
from configs.dataset_params import normalize_params
from demonstrator.tree_generator import TreeGenerator
from demonstrator.feature_grid_generator import FeatureGridGenerator


# CONSTANTS
N_TOTAL_FEATURES = 50
N_FEATURES_PER_CLASS = 5
IMG_SIZE = 224

# Global variables
CALIBRATION_DATA = None
TRAIN_LOADER = None
MODEL = None
FOLDER = None
DEVICE = None
CLASS_NAMES = None
global_k_predictions = 3
TREE_GENERATOR = None
FEATURE_GRID_GENERATOR = None

def initialize_demo():
    """Load model and calibration data once at startup"""
    global CALIBRATION_DATA, TRAIN_LOADER, MODEL, FOLDER, DEVICE, CLASS_NAMES, TREE_GENERATOR, FEATURE_GRID_GENERATOR
    
    args = get_args_for_loading_model()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL, FOLDER = load_model(args.dataset, args.arch, args.seed, args.model_type, 
                               args.cropGT, args.n_features, args.n_per_class, 
                               args.img_size, args.reduced_strides, args.folder)
    MODEL = MODEL.to(DEVICE)
    
    cal_path = Path.home() / "tmp" / "CHiQPM_calibration" / f"seed_{args.seed}" / "calibration_data.pt"
    CALIBRATION_DATA = torch.load(cal_path)
    print(f"Loaded calibration data: {len(CALIBRATION_DATA['cal_logits'])} samples")
    
    TRAIN_LOADER, _ = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    
    class_mapping = load_cub_class_mapping()
    CLASS_NAMES = {int(k): v for k, v in class_mapping.items()}
    
    TREE_GENERATOR = TreeGenerator(MODEL, CALIBRATION_DATA, CLASS_NAMES)
    
    FEATURE_GRID_GENERATOR = FeatureGridGenerator(
        MODEL, TRAIN_LOADER, FOLDER, CLASS_NAMES, DEVICE, 
        img_size=IMG_SIZE, n_features_per_class=N_FEATURES_PER_CLASS
    )

def forward_pass(input_image_pil):
    """Run forward pass once and return all needed outputs"""
    if input_image_pil is None:
        return None, None, None, None, None
    
    input_image, input_image_normalized = FEATURE_GRID_GENERATOR.preprocess_image(input_image_pil)
    
    with torch.no_grad():
        output_logits, feature_maps, final_features = MODEL(
            input_image_normalized, 
            with_feature_maps=True, 
            with_final_features=True
        )
    
    prediction = torch.argmax(output_logits).item()
    
    return input_image, input_image_normalized, output_logits, final_features, prediction

def generate_tree_from_outputs(output_logits, final_features, prediction, accuracy, colormapping):
    """Generate tree from already computed forward pass outputs"""
    return TREE_GENERATOR.generate_tree(output_logits, final_features, accuracy, colormapping)

def process_input_image(input_image_pil):
    """Initial processing when input image is uploaded - runs forward pass and generates everything with default accuracy 0.9"""
    if input_image_pil is None:
        return None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None, None, None, None
    
    input_image, input_image_normalized, output_logits, final_features, prediction = forward_pass(input_image_pil)
    predicted_class_name = CLASS_NAMES.get(prediction, f"Class {prediction}")
    
    viz_grid, colormapping = FEATURE_GRID_GENERATOR.generate_feature_grid(
        input_image, 
        input_image_normalized, 
        output_logits, 
        final_features,
        global_k_predictions
    )
    
    accuracy_default = 0.9
    tree_image = generate_tree_from_outputs(output_logits, final_features, prediction, accuracy=accuracy_default, colormapping=colormapping)
    
    return (
        output_logits,
        final_features,
        prediction,
        predicted_class_name,
        colormapping,
        accuracy_default,
        viz_grid,
        tree_image,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )

def classify_with_k_predictions(input_image_pil, k_value):
    """Update k_predictions and run classification"""
    global global_k_predictions
    global_k_predictions = int(k_value)
    
    return process_input_image(input_image_pil)

if __name__ == '__main__':
    print("Initializing demo...")
    initialize_demo()
    print("Demo ready!")
    
    with gr.Blocks() as demo:
        gr.Markdown("### CHiQPM - Interpretable Bird Classificator")
        
        # Always visible
        with gr.Row() as input_display_row:
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil",
                                       label="Upload Bird Image",
                                       height=200, 
                                       show_download_button=False
                                       )
            with gr.Column(scale=3):
                with gr.Row():
                    k_predictions_input = gr.Number(
                        value=3, 
                        label="Top k Predicted Classes", 
                        minimum=1, 
                        maximum=5, 
                        step=1,
                        precision=0
                    )
                    classify_btn = gr.Button("Classify", variant="secondary")
                predicted_class_name = gr.Textbox(label="Predicted Species", interactive=False)
        
        # Hidden until image uploaded - feature visualization
        with gr.Column(visible=False) as grid_section:
            gr.Markdown("#### Activated Features")
            comparison_grid = gr.Image(type="pil",
                                       show_download_button=False,
                                       show_share_button=False, 
                                       show_label=False,
                                       interactive=False,
                                       show_fullscreen_button=True
                                       )
        
        # Hidden until image uploaded - tree controls
        with gr.Column(visible=False) as tree_controls:
            gr.Markdown("#### Decision Tree")
            accuracy_slider = gr.Slider(
                minimum=0.8,
                maximum=0.999,
                value=0.9,
                step=0.001,
                label="Accuracy"
            )
            generate_tree_btn = gr.Button("Regenerate Tree with New Accuracy")
        
        # Hidden until image uploaded - tree display
        with gr.Column(visible=False) as tree_visualization:
            tree = gr.Image(type="pil", 
                            show_download_button=False, 
                            show_share_button=False, 
                            show_label=False, 
                            height=600,
                            interactive=False,
                            show_fullscreen_button=True
                            )
        
        # Hidden state variables to store forward pass outputs
        output_logits = gr.State(None)
        final_features = gr.State(None)
        prediction = gr.State(None)
        colormapping = gr.State(None)
        
        # Upload triggers everything with default accuracy
        classify_btn.click(
            fn=classify_with_k_predictions,
            inputs=[input_image, k_predictions_input],
            outputs=[
                output_logits,
                final_features,
                prediction,
                predicted_class_name,
                colormapping,
                accuracy_slider,
                comparison_grid,
                tree,
                grid_section,
                tree_controls,
                tree_visualization,
            ]
        )
        
        # Button regenerates tree only (no forward pass)
        generate_tree_btn.click(
            fn=generate_tree_from_outputs,
            inputs=[output_logits, final_features, prediction, accuracy_slider, colormapping],
            outputs=tree
        )

    demo.launch()