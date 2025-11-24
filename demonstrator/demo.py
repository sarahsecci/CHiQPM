import gradio as gr
import torch
from evaluation.load_model import get_args_for_loading_model, load_model
from dataset_classes.cub200 import load_cub_class_mapping
from get_data import get_data
from demonstrator.tree_generator import TreeGenerator
from demonstrator.feature_grid_generator import FeatureGridGenerator
from configs.demo_paths import CAL_DATA_ROOT

# CONSTANTS
N_FEATURES_PER_CLASS = 5
IMG_SIZE = 224
ACCURACY_STEP = 0.001
UPPER_BOUND = 0.999999
DEFAULT_ACCURACY_INDEX = 2
DEFAULT_K_PREDICTIONS = 3

# Global variables
CALIBRATION_DATA = None
TRAIN_LOADER = None
MODEL = None
FOLDER = None
DEVICE = None
CLASS_NAMES = None
TREE_GENERATOR = None
FEATURE_GRID_GENERATOR = None

def initialize_demo():
    """Load model and calibration data once at startup"""
    global CALIBRATION_DATA, TRAIN_LOADER, MODEL, FOLDER, DEVICE, CLASS_NAMES, TREE_GENERATOR, FEATURE_GRID_GENERATOR, TOTAL_ACCS
    
    args = get_args_for_loading_model()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL, FOLDER = load_model(args.dataset, args.arch, args.seed, args.model_type, 
                               args.cropGT, args.n_features, args.n_per_class, 
                               args.img_size, args.reduced_strides, args.folder)
    MODEL = MODEL.to(DEVICE)
    
    cal_path = CAL_DATA_ROOT / f"seed_{args.seed}" / "calibration_data.pt"
    CALIBRATION_DATA = torch.load(cal_path)
    print(f"Loaded calibration data: {len(CALIBRATION_DATA['cal_logits'])} samples")
    
    TRAIN_LOADER, _ = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    
    class_mapping = load_cub_class_mapping()
    CLASS_NAMES = {int(k): v for k, v in class_mapping.items()}
    
    TREE_GENERATOR = TreeGenerator(MODEL, CALIBRATION_DATA, CLASS_NAMES)
    TOTAL_ACCS = TREE_GENERATOR.predictor.score_function.total_accs.cpu().tolist()
    
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

def generate_tree_from_outputs(output_logits, final_features, accuracy, colormapping):
    """Generate tree from already computed forward pass outputs"""
    return TREE_GENERATOR.generate_tree(output_logits, final_features, accuracy, colormapping)

def process_input_image(input_image_pil, k_predictions):
    """Initial processing when input image is uploaded - runs forward pass and generates everything with default accuracy"""
    if input_image_pil is None:
        empty_button_updates = [gr.update(variant="secondary") for _ in range(len(TOTAL_ACCS))]
        return (None, None, None, None, None, None, None, gr.update(visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, gr.update(value=DEFAULT_K_PREDICTIONS), *empty_button_updates)
     
    input_image, input_image_normalized, output_logits, final_features, prediction = forward_pass(input_image_pil)
    predicted_class_name = CLASS_NAMES.get(prediction, f"Class {prediction}")
    
    viz_grid, colormapping = FEATURE_GRID_GENERATOR.generate_feature_grid(
        input_image, 
        input_image_normalized, 
        output_logits, 
        final_features,
        k_predictions
    )
    
    accuracy_default = TOTAL_ACCS[DEFAULT_ACCURACY_INDEX]
    tree_image = generate_tree_from_outputs(output_logits, final_features, accuracy=accuracy_default, colormapping=colormapping)
    button_updates = update_button_highlights(accuracy_default)

    return (
        input_image,
        input_image_normalized,
        output_logits,
        final_features,
        prediction,
        predicted_class_name,
        colormapping,
        gr.update(value=accuracy_default, minimum=TOTAL_ACCS[4], maximum=UPPER_BOUND),
        viz_grid,
        tree_image,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        k_predictions,
        gr.update(value=DEFAULT_K_PREDICTIONS),
        *button_updates,
    )

def update_button_highlights(current_accuracy):
    """Return button variant updates based on current accuracy"""
    updates = []
    for i in range(len(TOTAL_ACCS) - 1, -1, -1):
        acc = TOTAL_ACCS[i]
        if acc <= current_accuracy:
            updates.append(gr.update(variant="primary"))
        else:
            updates.append(gr.update(variant="secondary"))
    return updates

def regenerate_feature_grid_only(input_image_pil, k_predictions, output_logits, final_features, input_unnorm, input_norm):
    """
    Regenerate only the feature grid without running forward pass.
    Used when user changes k_predictions but image stays the same.
    
    Args:
        input_image_pil: Original PIL image (for checking if it changed)
        k_predictions: Number of top k predictions to show
        output_logits: Cached logits from previous forward pass
        final_features: Cached features from previous forward pass
        input_unnorm: Cached unnormalized input tensor
        input_norm: Cached normalized input tensor
        
    Returns:
        Updated feature grid image and colormapping
    """
    if input_image_pil is None or output_logits is None:
        return None, None
    
    viz_grid, colormapping = FEATURE_GRID_GENERATOR.generate_feature_grid(
        input_unnorm, 
        input_norm, 
        output_logits, 
        final_features,
        k_predictions
    )
    
    return viz_grid, colormapping

if __name__ == '__main__':
    print("Initializing demo...")
    initialize_demo()
    print("Demo ready!")
    
    with gr.Blocks(css=".reset-button { display: none !important; }") as demo:
        gr.Markdown("### CHiQPM - Interpretable Bird Classifier")
        
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
                        value=DEFAULT_K_PREDICTIONS, 
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
                minimum=TOTAL_ACCS[4],
                maximum=UPPER_BOUND,
                value=TOTAL_ACCS[DEFAULT_ACCURACY_INDEX],
                step=ACCURACY_STEP,
                label="Accuracy",
                show_label=True,
                interactive=True
            )
            
            gr.Markdown("**Quick select hierarchy levels:**")
            with gr.Row():
                level_buttons = []
                button_components = []
                for i in range(len(TOTAL_ACCS) - 1, -1, -1):
                    acc = TOTAL_ACCS[i]
                    N_number = len(TOTAL_ACCS) - i  # FIX: N=1 for i=4, N=5 for i=0
                    btn = gr.Button(f"N={N_number} ({acc:.3f})", size="sm", variant="secondary")
                    level_buttons.append((btn, acc))
                    button_components.append(btn)
                    
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
        input_unnormalized = gr.State(None)
        input_normalized = gr.State(None)  
        output_logits = gr.State(None)
        final_features = gr.State(None)
        prediction = gr.State(None)
        colormapping = gr.State(None)
        k_predictions = gr.State(DEFAULT_K_PREDICTIONS)

        for idx, (btn, acc_value) in enumerate(level_buttons):
            btn.click(
                fn=lambda acc, logits, feats, cmap: (
                    acc, 
                    generate_tree_from_outputs(logits, feats, acc, cmap),
                    *update_button_highlights(acc)
                ),
                inputs=[gr.State(acc_value), output_logits, final_features, colormapping],
                outputs=[accuracy_slider, tree] + button_components
            )
        
        # Upload triggers everything with default accuracy
        classify_btn.click(
            fn=process_input_image,
            inputs=[input_image, k_predictions_input],
            outputs=[
                input_unnormalized,
                input_normalized,  
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
                k_predictions,
                k_predictions_input,
                *button_components,
            ]
        )

        k_predictions_input.change(
            fn=regenerate_feature_grid_only,
            inputs=[input_image, k_predictions_input, output_logits, final_features, input_unnormalized, input_normalized],
            outputs=[comparison_grid, colormapping]
        )

        accuracy_slider.release(
            fn=lambda acc, logits, feats, cmap: (
                generate_tree_from_outputs(logits, feats, acc, cmap),
                *update_button_highlights(acc)
            ),
            inputs=[accuracy_slider, output_logits, final_features, colormapping],
            outputs=[tree] + button_components
        )

    demo.launch()