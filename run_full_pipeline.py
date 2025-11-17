import subprocess
import sys
import os
import argparse
from datetime import datetime
# Assuming src.config is in a reachable path or defined in the notebook
# If not, you may need to define BRSET_LABELS manually, e.g.:
# BRSET_LABELS = ['DME', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH', 'MRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCA', 'BPR', 'OS', 'SCH', 'SD']
from src.config import BRSET_LABELS

# -----------------------------------------------------------------
# --- DYNAMIC CONFIGURATION (via command-line) ---
# -----------------------------------------------------------------

# MODIFIED: Added 'args_list=None'
def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description="Full Training Pipeline for MoE Model")
    
    # --- Paths ---
    parser.add_argument('--labels-path', type=str, 
                        default='/kaggle/working/labels_brset.csv',
                        help="Path to the master labels.csv file")
    parser.add_argument('--image-dir', type=str, 
                        default='/kaggle/input/brset-retina-fundus-diagnose-image-dataset/data/fundus_photos', 
                        help="Path to the directory containing all images")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints", help="Root directory to save all checkpoints")
    
    # --- MLflow Config ---
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5000', help="MLflow tracking server URI")
    parser.add_argument('--mlflow-experiment', type=str, default='Retina MoE Project', help="MLflow experiment name")
    
    # --- Model Config (Global) ---
    parser.add_argument('--gate-model-name', type=str, default='resnet', help="Model name for the Gate")
    parser.add_argument('--gate-model-size', type=str, default='small', help="Model size for the Gate")
    parser.add_argument('--expert-model-name', type=str, default='resnet', help="Model name for the Experts")
    parser.add_argument('--expert-model-size', type=str, default='small', help="Model size for the Experts")
    
    # --- LoRA Config (Global) ---
    parser.add_argument('--use-lora', action='store_true', help="Use LoRA for both Gate and Experts")
    parser.add_argument('--use-qlora', action='store_true', help="Use Q-LoRA for both Gate and Experts")
    parser.add_argument('--lora-r', type=int, default=16, help="LoRA rank")
    
    # --- Training Params (Global) ---
    parser.add_argument('--gate-epochs', type=int, default=25, help="Epochs for Gate training")
    parser.add_argument('--expert-epochs', type=int, default=25, help="Epochs for Expert training")
    parser.add_argument('--batch-size', type=int, default=512, help="Batch size for Gate/Expert training")
    parser.add_argument('--base-lr', type=float, default=1e-4, help="Learning rate for Gate/Expert training")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    
    # --- Calibration Params ---
    parser.add_argument('--calibrate-epochs', type=int, default=3, help="Epochs for MoE calibration")
    parser.add_argument('--calibrate-batch-size', type=int, default=512, help="Batch size for calibration")
    parser.add_argument('--calibrate-lr', type=float, default=1e-6, help="Learning rate for calibration")

    # --- Evaluation Params ---
    parser.add_argument('--eval-batch-size', type=int, default=64, help="Batch size for final evaluation")

    # --- System Params ---
    parser.add_argument('--num-workers', type=int, default=4, help="Number of dataloader workers")
    
    # MODIFIED: Pass 'args_list' to parse_known_args
    # If args_list is None, it defaults to using sys.argv (command-line)
    args, _ = parser.parse_known_args(args_list)
    
    args.PATHOLOGIES = BRSET_LABELS
    args.GATE_CKPT_DIR = os.path.join(args.checkpoint_dir, 'gate')
    args.EXPERT_CKPT_DIR = os.path.join(args.checkpoint_dir, 'experts')
    args.FINAL_MOE_DIR = os.path.join(args.checkpoint_dir, 'final_moe')
    args.FINAL_MOE_PATH = os.path.join(args.FINAL_MOE_DIR, 'moe_calibrated_final.pth')

    return args

# -----------------------------------------------------------------
# --- HELPER FUNCTIONS ---
# -----------------------------------------------------------------

def run_command(cmd, log_file):
    # ... (function is unchanged) ...
    cmd_str = ' '.join(cmd)
    print(f"\n[RUNNING]: {cmd_str}")
    log_file.write(f"\n[RUNNING]: {cmd_str}\n")
    log_file.flush()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        sys.stdout.write(line)
        log_file.write(line)
        log_file.flush()
    return_code = process.wait()
    if return_code != 0:
        error_msg = f"\n*** ERROR ***: Command failed with return code {return_code}."
        print(error_msg, file=sys.stderr)
        log_file.write(error_msg + "\n")
        raise subprocess.CalledProcessError(return_code, cmd_str)
    print(f"\n[SUCCESS]: Command finished successfully.")
    log_file.write(f"\n[SUCCESS]: Command finished successfully.\n")

# -----------------------------------------------------------------
# --- PIPELINE PHASES ---
# -----------------------------------------------------------------

def run_phase_0_build_cache(args, log_file):
    # ... (function is unchanged) ...
    print("\n" + "="*80)
    print("--- PHASE 0: BUILDING IMAGE CACHE (if needed) ---")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n--- PHASE 0: BUILDING IMAGE CACHE (if needed) ---\n" + "="*80 + "\n")
    cmd = [
        'python', 'src/0_build_cache.py',
        '--labels-path', args.labels_path,
        '--image-dir', args.image_dir,
    ]
    run_command(cmd, log_file)

def run_phase_1_gate(args, log_file):
    """Runs the 1_train_gate.py script."""
    print("\n" + "="*80)
    print("--- PHASE 1: TRAINING GATE MODEL ---")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n--- PHASE 1: TRAINING GATE MODEL ---\n" + "="*80 + "\n")

    cmd = [
        'python', 'src/1_train_gate.py',
        '--labels-path', args.labels_path,
        '--image-dir', args.image_dir,
        '--output-dir', args.GATE_CKPT_DIR,
        '--run-name', 'Phase_1_Gate_Model', # <-- MLflow Run Name
        '--model-name', args.gate_model_name,
        '--model-size', args.gate_model_size,
        '--epochs', str(args.gate_epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.base_lr),
        '--patience', str(args.patience),
        '--num-workers', str(args.num_workers),
        '--lora-r', str(args.lora_r),
    ]
    if args.use_lora: cmd.append('--use-lora')
    if args.use_qlora: cmd.append('--use-qlora')
    if args.use_mlflow: cmd.append('--use-mlflow') # <-- Pass MLflow flag
    run_command(cmd, log_file)

def run_phase_2_experts(args, log_file):
    """Runs the 2_train_experts.py script for all pathologies."""
    print("\n" + "="*80)
    print("--- PHASE 2: TRAINING EXPERT MODELS ---")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n--- PHASE 2: TRAINING EXPERT MODELS ---\n" + "="*80 + "\n")

    for i, pathology in enumerate(args.PATHOLOGIES):
        print(f"\n--- Expert {i+1}/{len(args.PATHOLOGIES)}: {pathology} ---")
        log_file.write(f"\n--- Expert {i+1}/{len(args.PATHOLOGIES)}: {pathology} ---\n")
        cmd = [
            'python', 'src/2_train_experts.py',
            '--labels-path', args.labels_path,
            '--image-dir', args.image_dir,
            '--target-label', pathology,
            '--output-dir', args.EXPERT_CKPT_DIR,
            '--run-name', f'Phase_2_Expert_{pathology}', # <-- MLflow Run Name
            '--model-name', args.expert_model_name,
            '--model-size', args.expert_model_size,
            '--epochs', str(args.expert_epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.base_lr),
            '--patience', str(args.patience),
            '--num-workers', str(args.num_workers),
            '--lora-r', str(args.lora_r),
        ]
        if args.use_lora: cmd.append('--use-lora')
        if args.use_qlora: cmd.append('--use-qlora')
        if args.use_mlflow: cmd.append('--use-mlflow') # <-- Pass MLflow flag
        run_command(cmd, log_file)

def run_phase_3_calibrate(args, log_file):
    """Runs the 3_calibrate_moe.py script."""
    print("\n" + "="*80)
    print("--- PHASE 3: CALIBRATING HYBRID MOE ---")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n--- PHASE 3: CALIBRATING HYBRID MOE ---\n" + "="*80 + "\n")

    cmd = [
        'python', 'src/3_calibrate_moe.py',
        '--labels-path', args.labels_path,
        '--image-dir', args.image_dir,
        '--output-dir', args.FINAL_MOE_DIR,
        '--run-name', 'Phase_3_MoE_Calibration', # <-- MLflow Run Name
        '--gate-ckpt-path', args.GATE_CKPT_DIR,
        '--expert-ckpt-dir', args.EXPERT_CKPT_DIR,
        '--gate-model-name', args.gate_model_name,
        '--gate-model-size', args.gate_model_size,
        '--expert-model-name', args.expert_model_name,
        '--expert-model-size', args.expert_model_size,
        '--lora-r', str(args.lora_r),
        '--epochs', str(args.calibrate_epochs),
        '--batch-size', str(args.calibrate_batch_size),
        '--lr', str(args.calibrate_lr),
        '--num-workers', str(args.num_workers),
    ]
    if args.use_lora:
        cmd.append('--gate-use-lora')
        cmd.append('--expert-use-lora')              
    if args.use_qlora:
        cmd.append('--gate-use-qlora')
        cmd.append('--expert-use-qlora')
    if args.use_mlflow: cmd.append('--use-mlflow') # <-- Pass MLflow flag
    run_command(cmd, log_file)

def run_phase_4_evaluate(args, log_file):
    """Runs the evaluate.py script."""
    print("\n" + "="*80)
    print("--- PHASE 4: FINAL EVALUATION ---")
    print("="*80)
    log_file.write("\n" + "="*80 + "\n--- PHASE 4: FINAL EVALUATION ---\n" + "="*80 + "\n")

    cmd = [
        'python', 'src/evaluate.py',
        '--labels-path', args.labels_path,
        '--image-dir', args.image_dir,
        '--model-path', args.FINAL_MOE_PATH,
        '--run-name', 'Phase_4_Final_MoE_Evaluation', # <-- MLflow Run Name
        '--gate-model-name', args.gate_model_name,
        '--gate-model-size', args.gate_model_size,
        '--expert-model-name', args.expert_model_name,
        '--expert-model-size', args.expert_model_size,
        '--batch-size', str(args.eval_batch_size),
        '--num-workers', str(args.num_workers),
        '--lora-r', str(args.lora_r),
    ]
    if args.use_lora:
        cmd.append('--gate-use-lora')
        cmd.append('--expert-use-lora')
    if args.use_qlora:
        cmd.append('--gate-use-qlora')
        cmd.append('--expert-use-qlora')
    if args.use_mlflow: cmd.append('--use-mlflow') # <-- Pass MLflow flag
    run_command(cmd, log_file)

# -----------------------------------------------------------------
# --- MAIN EXECUTION ---
# -----------------------------------------------------------------

# MODIFIED: Added 'args_list=None'
def main(args_list=None):
    # MODIFIED: Pass 'args_list'
    args = parse_arguments(args_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pipeline_log_{timestamp}.txt"
    print(f"Starting full pipeline. Log will be saved to: {log_filename}")
    print(f"Using {len(args.PATHOLOGIES)} pathologies: {args.PATHOLOGIES}")
    
    if args.use_mlflow:
        print(f"\n--- MLFLOW IS ENABLED ---")
        print(f"Tracking URI: {args.mlflow_uri}")
        print(f"Experiment:   {args.mlflow_experiment}")
        os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_uri
        os.environ['MLFLOW_EXPERIMENT_NAME'] = args.mlflow_experiment
        # NOTE: Set S3/MinIO env vars in your notebook/shell environment
        # os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
        # os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
        # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
        
        # Try to import mlflow to catch errors early
        try:
            import mlflow
            print("MLflow client imported successfully.")
        except ImportError:
            print("\n*** ERROR ***: --use-mlflow was specified, but 'mlflow' package is not found.")
            print("Please install it: pip install mlflow")
            sys.exit(1)
    
    try:
        with open(log_filename, 'w') as log_file:
            log_file.write(f"--- Full Pipeline Log - {timestamp} ---\n")
            log_file.write(f"Running with arguments: {vars(args)}\n")
            
            run_phase_0_build_cache(args, log_file)
            run_phase_1_gate(args, log_file)
            run_phase_2_experts(args, log_file)
            run_phase_3_calibrate(args, log_file)
            run_phase_4_evaluate(args, log_file)

            print(f"\nPIPELINE COMPLETED SUCCESSFULLY! Log saved to: {log_filename}")
            log_file.write("\nPIPELINE COMPLETED SUCCESSFULLY!\n")

    except subprocess.CalledProcessError:
        print(f"\nPIPELINE FAILED. Check log for details: {log_filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        print(f"PIPELINE FAILED. Check log for details: {log_filename}", file=sys.stderr)
        sys.exit(1)

# -----------------------------------------------------------------
# --- MODIFIED: Notebook-friendly execution ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    # --- V V V --- EDIT YOUR ARGUMENTS HERE --- V V V ---
    #
    # Define your arguments as a list of strings, just as you would
    # type them on the command line.
    #
    notebook_args = [
        '--labels-path', '/kaggle/working/my_labels.csv',
        '--image-dir', '/kaggle/input/my-image-dataset/images',
        '--checkpoint-dir', '/kaggle/working/checkpoints',
        '--use-mlflow',
        '--gate-epochs', '10',
        '--expert-epochs', '10',
        '--batch-size', '256',
        '--use-lora'  # <-- Explicitly adding this flag
        # '--use-qlora' # <-- Uncomment to use QLoRA
        # '--lora-r', '32'  # <-- Example of overriding a default
    ]
    #
    # --- ^ ^ ^ --- EDIT YOUR ARGUMENTS HERE --- ^ ^ ^ ---


    # To run with the script's built-in defaults, use an empty list:
    # notebook_args = []
    
    # To run from the command line, this block is skipped and
    # 'main()' would be called with 'args_list=None',
    # which defaults to using sys.argv.
    
    print(f"--- Running pipeline in NOTEBOOK mode with arguments: ---")
    print(notebook_args)
    print("---------------------------------------------------------")

    try:
        # Pass the list of arguments to main()
        main(notebook_args)
        
    except SystemExit as e:
        # This block catches sys.exit() calls
        if e.code == 0:
            print("\nPipeline finished successfully (caught SystemExit 0).")
        else:
            print(f"\nPIPELINE FAILED (caught SystemExit code: {e.code})", file=sys.stderr)
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)