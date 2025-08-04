# ADATime/integrative_benchmarking_app.py
import streamlit as st
import re
import os
import torch
import wandb
import argparse
import pandas as pd
from datetime import datetime
from Algorithm2Domain-ADATime.trainers.sweep import Trainer as SweepTrainer
from Algorithm2Domain-ADATime.trainers.train import Trainer as TrainTrainer
from Algorithm2Domain-ADATime.configs.data_model_configs import *
from Algorithm2Domain-ADATime.configs.sweep_params import sweep_alg_hparams, get_combined_sweep_hparams, get_sweep_train_hparams
from Algorithm2Domain-ADATime.configs.data_model_configs import get_dataset_class

torch.classes.__path__ = []

# ----------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------

# Setting the location for saving the wandb files
os.environ['WANDB_DIR'] = 'Algorithm2Domain-ADATime'

st.set_page_config(
    page_title="Algorithm2Domain - Integrative Benchmarking",
    page_icon=":material/schedule:",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ----------------------------------------------------------------------
# 2. CSS 
# ----------------------------------------------------------------------

def load_css(file_name: str):
    """Load local CSS for custom styling."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 3. Header
# ----------------------------------------------------------------------
def header():
    """Display the header of the app."""
    header_html = """
    <div class="header">
        <h1>Algorithm2Domain</h1>
        <h3>Integrative Benchmarking Platform</h3>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")


# ----------------------------------------------------------------------
# 4. Form Data
# ----------------------------------------------------------------------

# Data for Evaluation Setup
# """
#     Scenarios - A Scenario is a Tuple of (Source, Target)
#         - Only Unique tuples are allowed.
#         - The source != target for all tuples.
#         - There can be more than one scenario.
#         - In string format. Eg: ("1", "2")
#
#     Experiment Name
#         - Name of the experiment
#
#     Dataset Choice
#         - Currently only single dataset is allowed.
#         - As cross-dataset evaluation is not supported yet.
#         - Radio buttons for the dataset choice.
#         - Current choices are:
#             - HHAR
#             - WEATHER
#             - OHIO
#             - Other
#
#     Number of Runs
#         - The number of runs for each scenario.
# """
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

if "experiment_name" not in st.session_state:
    st.session_state.experiment_name = ""

if "dataset_choice" not in st.session_state:
    st.session_state.dataset_choice = ""

if "number_of_runs" not in st.session_state:
    st.session_state.number_of_runs = 1

# Data for Hyperparameter Tuning
# """
#     Hyperparameter Tuning
#         - Radio buttons for the hyperparameter tuning.
#             - No Hyperparameter Tuning
#             - Hyperparameter Tuning
# """
if "hparams_hyperparameter_tuning" not in st.session_state:
    st.session_state.hparams_hyperparameter_tuning = "No Hyperparameter Tuning"

if "hparams_num_sweeps" not in st.session_state:
    st.session_state.hparams_num_sweeps = 1

if "hparams_sweep_project_name" not in st.session_state:
    st.session_state.hparams_sweep_project_name = ""

if "hparams_target_risk_score" not in st.session_state:
    st.session_state.hparams_target_risk_score = "dev_risk"

if "hparams_num_epochs" not in st.session_state:
    st.session_state.hparams_num_epochs = [3, 4, 5, 6]

if "hparams_batch_size" not in st.session_state:
    st.session_state.hparams_batch_size = [32, 64]

if "hparams_learning_rate" not in st.session_state:
    st.session_state.hparams_learning_rate = [1e-2, 5e-3, 1e-3, 5e-4]

if "hparams_disc_lr" not in st.session_state:
    st.session_state.hparams_disc_lr = [1e-2, 5e-3, 1e-3, 5e-4]

if "hparams_weight_decay" not in st.session_state:
    st.session_state.hparams_weight_decay = [1e-4, 1e-5, 1e-6]

if "hparams_step_size" not in st.session_state:
    st.session_state.hparams_step_size = [5, 10, 30]

if "hparams_gamma" not in st.session_state:
    st.session_state.hparams_gamma = [5, 10, 15, 20, 25]

if "hparams_optimizer" not in st.session_state:
    st.session_state.hparams_optimizer = ["adam"]

# Data for Data Augmentation
# """
#     Preprocessing
#         - Normalization
#             - Currently only 2 radio buttons.
#                 - Standard Normalization
#                 - No Normalization
# """
if "normalization" not in st.session_state:
    st.session_state.normalization = ""


# Data for Backbone Model
# """
#     Backbone Model
#         - Drop down menu containing the following options:
#             - Select Model (default)
#             - CNN (Convolutional Neural Network)
#             - RESNET (Residual Network)
#             - TCN (Temporal Convolutional Network)
#
#     Hyperparameters
#         - Input text fields for the following hyperparameters:
#             - Kernel Size
#             - Stride
#             - Dropout
#         
#     Only when the model is selected, the following fields will be shown.
#     Model Specific Parameters
#         - Input text fields for the following hyperparameters:
#             - Mid Channels
#             - Final Outcome Channels
#             - Features Length
# """
if "backbone_model" not in st.session_state:
    st.session_state.backbone_model = "Select Model"

if "kernel_size" not in st.session_state:
    st.session_state.kernel_size = 5

if "stride" not in st.session_state:
    st.session_state.stride = 1

if "dropout" not in st.session_state:
    st.session_state.dropout = 0.5

if "mid_channels" not in st.session_state:
    st.session_state.mid_channels = 64

if "final_outcome_channels" not in st.session_state:
    st.session_state.final_outcome_channels = 2

if "features_length" not in st.session_state:
    st.session_state.features_length = 32


# Data for Training Strategy
# """
#     Algorithm
#         - Drop down menu containing the following options:
#             - Select Algorithm (default)
#             - NO_ADAPT
#             - Deep_Coral
#             - MMDA
#             - DANN
#             - CDAN
#             - DIRT
#             - DSAN
#             - HoMM
#             - CoDATS
#             - AdvSKM
#             - SASA
#             - CoTMix
#             - TARGET_ONLY
#         
#     Training Parameters
#         - Input text fields for the following hyperparameters:
#             - Number of Epochs
#             - Batch Size
#             - Weight Decay
#             - Step Size (optional)
#             - Learning Rate Decay (optional)
#
#     Only when the algorithm is selected, the following fields will be shown.
#     Algorithm Specific Parameters
#         - Input text fields for the following hyperparameters:
# """
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "Select Algorithm"

if "number_of_epochs" not in st.session_state:
    st.session_state.number_of_epochs = 10

if "batch_size" not in st.session_state:
    st.session_state.batch_size = 32

if "weight_decay" not in st.session_state:
    st.session_state.weight_decay = 0.0001

if "step_size" not in st.session_state:
    st.session_state.step_size = 10

if "lr_decay" not in st.session_state:
    st.session_state.lr_decay = 0.1


# Data for Results
# """
#     Results
# """
#if "results" not in st.session_state:
#    st.session_state.results = []



# ----------------------------------------------------------------------
# 5. Evaluation Setup
# ----------------------------------------------------------------------

def evaluation_setup():
    """
    Evaluation setup
        - This function will be used to setup the evaluation.
        - It will contain the following fields:
            - Scenarios
            - Experiment Name
            - Dataset Choice
            - Number of Runs
    """

    with st.expander("Evaluation Configuration", expanded=True):
        st.subheader("Evaluation Setup")
        
        # Experiment Name
        st.session_state.experiment_name = st.text_input(
            "Experiment Name",
            placeholder="Eg: EXP1",
            value=st.session_state.experiment_name,
            help="Enter the name of your experiment."
        )

        # Can you add to automatically detect the datasets in the folder?
        # Dataset Path
        dataset_path = "ADATIME_data/"
        # Retrieve the list of datasets in the folder
        dataset_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        st.write(dataset_list)
        
        # Dataset Choice (Radio Buttons)
        if dataset_list == []:
            #dataset_options = ["HHAR", "WEATHER", "OHIO", "FD", "HAR", "EEG", "WISDM"]
            dataset_options = ["HHAR", "WEATHER", "OHIO"]
        else:
            dataset_options = dataset_list
        
        st.session_state.dataset_choice = st.radio(
            "Dataset Choice",
            options=dataset_options,
            index=dataset_options.index(st.session_state.dataset_choice)
                if st.session_state.dataset_choice in dataset_options else 0,
            help="Select a single dataset for the evaluation."
        )
        
        st.write("---")

        dataset_path = f"ADATIME_data/{st.session_state.dataset_choice}"

        try:
            files = os.listdir(dataset_path)
            train_files = [f for f in files if re.match(r"train_\d+\.pt", f)]
            domain_indices = sorted({int(re.findall(r"\d+", f)[0]) for f in train_files})
            domain_options = [str(i) for i in domain_indices]
        except Exception as e:
            st.error(f"Error loading domain files: {e}")
            domain_options = ["0"]
        
        # Scenarios (Add multiple source-target pairs)
        st.write("**Scenarios**")
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source Domain", options=domain_options, index=0)
        with col2:
            target = st.selectbox("Target Domain", options=domain_options, index=1)
        
        # Button to add a scenario
        if st.button("➕ Add Scenario", use_container_width=True):
            if source == target:
                st.error("Source and Target must be different!")
            else:
                new_scenario = (str(source), str(target))
                if new_scenario not in st.session_state.scenarios:
                    st.session_state.scenarios.append(new_scenario)
                    st.success(f"Scenario added: {new_scenario}")
                else:
                    st.warning("This scenario has already been added.")
        
        # Display configured scenarios
        if st.session_state.scenarios:
            st.markdown("**Configured Scenarios:**")
            for idx, scenario in enumerate(st.session_state.scenarios, start=1):
                st.write(f"{idx}. {scenario}")
        else:
            st.info("No scenarios added yet. Please add at least one scenario.")
        
        st.write("---")
        
        # Number of Runs per Scenario
        st.session_state.number_of_runs = st.number_input(
            "Number of Runs per Scenario",
            min_value=1,
            max_value=100,
            value=st.session_state.number_of_runs,
            help="Define the number of runs for each configured scenario."
        )
        
        # Validate the evaluation setup
        if st.button("Validate Evaluation Setup", use_container_width=True):
            # Check if each field is filled
            # Experiment Name
            if not st.session_state.experiment_name.strip():
                st.error("Experiment Name cannot be empty!")
            # Dataset Choice
            elif not st.session_state.dataset_choice:
                st.error("Please select a Dataset!")
            # Scenarios
            elif not st.session_state.scenarios:
                st.error("Please add at least one Scenario!")
            # Number of Runs
            elif st.session_state.number_of_runs < 1:
                st.error("Number of Runs must be at least 1!")
            # If all fields are valid
            else:
                # Save the evaluation setup to session state
                st.session_state.evaluation_setup = {
                    "experiment_name": st.session_state.experiment_name,
                    "dataset_choice": st.session_state.dataset_choice,
                    "scenarios": st.session_state.scenarios,
                    "number_of_runs": st.session_state.number_of_runs
                }
                st.json(st.session_state.evaluation_setup, expanded=True)
                st.session_state.evaluation_setup_valid = True
                st.success("Evaluation Setup is valid! You can proceed to the next step.")


# ----------------------------------------------------------------------
# 6. Hyperparameter Tuning
# ----------------------------------------------------------------------

def hyperparameter_tuning():
    """
    Hyperparameter Tuning
        - This function configures hyperparameter tuning options.
        - It provides two modes:
            (1) No Hyperparameter Tuning, or 
            (2) Hyperparameter Tuning.
        - When tuning is enabled, the user must supply:
            - Sweep Project Name
            - Number of Sweeps
            - Target Risk Score to Optimize
            - Training Parameters:
                - Number of Epochs (Integer)
                - Batch Size (Integer)
                - Learning Rate (Float)
                - Discriminator Learning Rate (Float)
                - Weight Decay (Float)
                - Step Size (Integer)
                - Gamma (Integer)
                - Optimizer (String)
        - For each training parameter, a multiselect is used to show predefined options (converted to strings)
          with an inline text input allowing comma-separated additional values.
        - The entered values are then combined, converted to the proper type, de-duplicated, sorted,
          and stored in session state.
    """
    with st.expander("Hyperparameters Configuration", expanded=True):
        st.subheader("Hyperparameter Tuning")
        
        # ------------------------------------------------------------------
        # Hyperparameter Tuning (Radio Buttons)
        # ------------------------------------------------------------------
        tuning_options = ["No Hyperparameter Tuning", "Hyperparameter Tuning"]
        st.session_state.hparams_hyperparameter_tuning = st.radio(
            "Hyperparameter Tuning",
            options=tuning_options,
            index=tuning_options.index(st.session_state.hparams_hyperparameter_tuning)
                    if st.session_state.get("hparams_hyperparameter_tuning") in tuning_options else 0,
            help="Select the hyperparameter tuning method."
        )
        
        if st.session_state.hparams_hyperparameter_tuning == "Hyperparameter Tuning":
            # -------------------------------
            # Sweep Project Name
            # -------------------------------
            st.session_state.hparams_sweep_project_name = st.text_input(
                "Sweep Project Name",
                placeholder="Eg: ADATime_Sweep",
                value=st.session_state.get("hparams_sweep_project_name", ""),
                help="Enter the name of your sweep project."
            )

            # -------------------------------
            # Sweep Project WandB
            # -------------------------------
            st.session_state.hparams_sweep_project_wandb = st.text_input(
                "Sweep Project WandB",
                placeholder="Eg: ADATime_Sweep_WandB",
                value=st.session_state.get("hparams_sweep_project_wandb", ""),
                help="Enter the name of your sweep project in WandB."
            )
            
            # -------------------------------
            # Number of Sweeps
            # -------------------------------
            st.session_state.hparams_num_sweeps = st.number_input(
                "Number of Sweeps",
                min_value=1,
                value=st.session_state.get("hparams_num_sweeps", 1),
                step=1,
                help="Define the number of sweeps."
            )
            
            # -------------------------------
            # Target Risk Score to Optimize
            # -------------------------------
            risk_score_options = ["dev_risk", "trg_risk", "src_risk", "few_shot_trg_risk"]
            st.session_state.hparams_target_risk_score = st.selectbox(
                "Target Risk Score to Optimize",
                options=risk_score_options,
                index=risk_score_options.index(st.session_state.get("hparams_target_risk_score", risk_score_options[0]))
                      if st.session_state.get("hparams_target_risk_score") in risk_score_options else 0,
                help="Select the target risk score to optimize."
            )
            
            # ------------------------------------------------------------------
            # Training Parameters with inline additional custom values
            # ------------------------------------------------------------------
            st.markdown("#### Training Parameters")
            
            # ---------------------
            # Number of Epochs (Integer)
            # ---------------------
            additional_str = st.text_input(
                "Number of Epochs (enter as comma separated list)",
                key="hparams_num_epochs_additional",
                help="Enter additional epoch values, separated by commas.",
                value="3, 4, 5, 6"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([int(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered epoch values are not valid integers.")
                combined_values = []
            st.session_state.hparams_num_epochs = combined_values
            
            # ---------------------
            # Batch Size (Integer)
            # ---------------------
            additional_str = st.text_input(
                "Batch Size (enter as comma separated list)",
                key="hparams_batch_size_additional",
                help="Enter additional batch size values, separated by commas.",
                value="32, 64"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([int(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered batch size values are not valid integers.")
                combined_values = []
            st.session_state.hparams_batch_size = combined_values
            
            # ---------------------
            # Learning Rate (Float)
            # ---------------------
            additional_str = st.text_input(
                "Learning Rate (enter as comma separated list)",
                key="hparams_learning_rate_additional",
                help="Enter additional learning rate values, separated by commas.",
                value="1e-2, 5e-3, 1e-3, 5e-4"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([float(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered learning rate values are not valid numbers.")
                combined_values = []
            st.session_state.hparams_learning_rate = combined_values
            
            # ---------------------
            # Discriminator Learning Rate (Float)
            # ---------------------
            additional_str = st.text_input(
                "Discriminator Learning Rate (enter as comma separated list)",
                key="hparams_disc_lr_additional",
                help="Enter additional discriminator learning rate values, separated by commas.",
                value="1e-2, 5e-3, 1e-3, 5e-4"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([float(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered discriminator learning rate values are not valid numbers.")
                combined_values = []
            st.session_state.hparams_disc_lr = combined_values
            
            # ---------------------
            # Weight Decay (Float)
            # ---------------------
            additional_str = st.text_input(
                "Weight Decay (enter as comma separated list)",
                key="hparams_weight_decay_additional",
                help="Enter additional weight decay values, separated by commas.",
                value="1e-4, 1e-5, 1e-6"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([float(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered weight decay values are not valid numbers.")
                combined_values = []
            st.session_state.hparams_weight_decay = combined_values

            # ---------------------
            # Step Size (Integer)
            # ---------------------
            additional_str = st.text_input(
                "Step Size (enter as comma separated list)",
                key="hparams_step_size_additional",
                help="Enter additional step size values, separated by commas.",
                value="5, 10, 30"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([int(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered step size values are not valid integers.")
                combined_values = []
            st.session_state.hparams_step_size = combined_values
            
            # ---------------------
            # Gamma (Integer)
            # ---------------------
            additional_str = st.text_input(
                "Gamma (enter as comma separated list)",
                key="hparams_gamma_additional",
                help="Enter additional gamma values, separated by commas.",
                value="5, 10, 15, 20, 25"
            )
            additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            try:
                combined_values = sorted(list(set([int(x) for x in (additional_values)])))
            except ValueError:
                st.error("One or more entered gamma values are not valid integers.")
                combined_values = []
            st.session_state.hparams_gamma = combined_values
            
            # # ---------------------
            # # Optimizer (String)
            # # with Multiselect
            # # ---------------------
            # # For now just make it immutable
            # default_optimizer = ["adam"]
            # default_values = default_optimizer  # already strings
            # selected_values = st.multiselect(
            #     label="Optimizer (Select one or more)",
            #     options=default_values,
            #     default=default_values,
            #     key="hparams_optimizer_multiselect",
            #     help="Select one or more optimizer names."
            # )
            # additional_str = st.text_input(
            #     "Add additional custom optimizer names (comma separated)",
            #     key="hparams_optimizer_additional",
            #     help="Enter additional optimizer names, separated by commas."
            # )
            # additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
            # combined_values = sorted(list(set(selected_values + additional_values)))
            # st.session_state.hparams_optimizer = combined_values

            # ---------------------
            # Optimizer (String)
            # without Multiselect
            # ---------------------
            # For now make it immutable
            st.markdown("<br>", unsafe_allow_html=True)
            default_optimizer = ["adam"]
            st.write("Optimizer: adam")
            st.session_state.hparams_optimizer = default_optimizer
            st.markdown("<br>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Validate the Hyperparameter Tuning Inputs
        # ------------------------------------------------------------------
        if st.button("Validate Hyperparameter Tuning", use_container_width=True):
            errors = []
            if st.session_state.hparams_hyperparameter_tuning == "Hyperparameter Tuning":
                if not st.session_state.hparams_sweep_project_name:
                    errors.append("Sweep Project Name cannot be empty.")
                if not st.session_state.hparams_num_sweeps or st.session_state.hparams_num_sweeps < 1:
                    errors.append("Number of Sweeps must be at least 1.")
                if not st.session_state.hparams_target_risk_score:
                    errors.append("Please select a target risk score.")
                if not st.session_state.hparams_num_epochs:
                    errors.append("Please select at least one value for Number of Epochs.")
                if not st.session_state.hparams_batch_size:
                    errors.append("Please select at least one value for Batch Size.")
                if not st.session_state.hparams_learning_rate:
                    errors.append("Please select at least one value for Learning Rate.")
                if not st.session_state.hparams_disc_lr:
                    errors.append("Please select at least one value for Discriminator Learning Rate.")
                if not st.session_state.hparams_weight_decay:
                    errors.append("Please select at least one value for Weight Decay.")
                if not st.session_state.hparams_step_size:
                    errors.append("Please select at least one value for Step Size.")
                if not st.session_state.hparams_gamma:
                    errors.append("Please select at least one value for Gamma.")
                if not st.session_state.hparams_optimizer:
                    errors.append("Please select at least one optimizer.")
            
            # Display errors if any exist, else save validation as successful
            if errors:
                for err in errors:
                    st.error(err)
                st.session_state.hyperparameter_tuning_valid = False
            else:
                st.session_state.hyperparameter_tuning_valid = True

                # Only unique values are saved to avoid duplicates
                st.session_state.hparams_num_epochs = list(set(st.session_state.hparams_num_epochs))
                st.session_state.hparams_batch_size = list(set(st.session_state.hparams_batch_size))
                st.session_state.hparams_learning_rate = list(set(st.session_state.hparams_learning_rate))
                st.session_state.hparams_disc_lr = list(set(st.session_state.hparams_disc_lr))
                st.session_state.hparams_weight_decay = list(set(st.session_state.hparams_weight_decay))
                st.session_state.hparams_step_size = list(set(st.session_state.hparams_step_size))
                st.session_state.hparams_gamma = list(set(st.session_state.hparams_gamma))
                st.session_state.hparams_optimizer = list(set(st.session_state.hparams_optimizer))

                if st.session_state.hparams_hyperparameter_tuning == "No Hyperparameter Tuning":
                    st.session_state.hyperparameter_tuning_config = {
                        "hyperparameter_tuning": st.session_state.hparams_hyperparameter_tuning,
                    }
                else:
                    # Save the hyperparameter tuning configuration to session state
                    st.session_state.hyperparameter_tuning_config = {
                        "hyperparameter_tuning": st.session_state.hparams_hyperparameter_tuning,
                        "sweep_project_name": st.session_state.hparams_sweep_project_name,
                        "sweep_project_wandb": st.session_state.hparams_sweep_project_wandb,
                        "num_sweeps": st.session_state.hparams_num_sweeps,
                        "target_risk_score": st.session_state.hparams_target_risk_score,
                        "num_epochs": {"values": st.session_state.hparams_num_epochs},
                        "batch_size": {"values": st.session_state.hparams_batch_size},
                        "learning_rate": {"values": st.session_state.hparams_learning_rate},
                        "disc_lr": {"values": st.session_state.hparams_disc_lr},
                        "weight_decay": {"values": st.session_state.hparams_weight_decay},
                        "step_size": {"values": st.session_state.hparams_step_size},
                        "gamma": {"values": st.session_state.hparams_gamma},
                        "optimizer": {"values": st.session_state.hparams_optimizer}
                    }

                st.json(st.session_state.hyperparameter_tuning_config, expanded=True)
                st.success("Hyperparameter Tuning settings are valid! You can proceed to the next step.")


# ----------------------------------------------------------------------
# 7. Data Augmentation
# ----------------------------------------------------------------------

def data_augmentation():
    """
    Data Augmentation
        - Configure one augmentation “option” each for source and target.
        - For each:
            • Add percentage of augmented samples (0.0–1.0)
            • Select any of:
                – MagnitudeWarping (σ)
                – GaussianNoise (mean, std)
                – WindowWarping (scales, window_ratio)
    """
    # initialize defaults
    if "da_source_pct" not in st.session_state:
        st.session_state.da_source_pct = 0.2
    if "da_source_steps" not in st.session_state:
        st.session_state.da_source_steps = []
    if "da_target_pct" not in st.session_state:
        st.session_state.da_target_pct = 0.2
    if "da_target_steps" not in st.session_state:
        st.session_state.da_target_steps = []

    with st.expander("Data Augmentation Configuration", expanded=True):
        st.subheader("Data Augmentation")

        # ----------------------------------------------------------------
        # Source Data Augmentation
        # ----------------------------------------------------------------
        st.markdown("#### Source Data Augmentation")
        st.session_state.da_source_pct = st.number_input(
            "Add % of augmented samples",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=st.session_state.da_source_pct,
            help="Fraction of new source samples to generate"
        )
        src_steps = []
        if st.checkbox("Magnitude Warping", key="src_mw"):
            sigma = st.number_input(
                "  MW σ",
                min_value=0.0,
                value=1.0,
                key="src_mw_sigma"
            )
            src_steps.append({"type": "MagnitudeWarping", "params": {"sigma": sigma}})
        if st.checkbox("Gaussian Noise", key="src_gn"):
            mean = st.number_input("  GN mean", value=0.0, key="src_gn_mean")
            std  = st.number_input("  GN std",  min_value=0.0, value=0.1, key="src_gn_std")
            src_steps.append({"type": "GaussianNoise", "params": {"mean": mean, "std": std}})
        if st.checkbox("Window Warping", key="src_ww"):
            scales_str = st.text_input("  WW scales (comma‑sep)", value="0.5", key="src_ww_scales")
            window_ratio = st.number_input("  WW window_ratio", min_value=0.0, max_value=1.0, value=0.5, key="src_ww_ratio")
            try:
                scales = [float(x.strip()) for x in scales_str.split(",") if x.strip()]
            except ValueError:
                st.error("WW scales must be numbers")
                scales = []
            src_steps.append({"type": "WindowWarping", "params": {"scales": scales, "window_ratio": window_ratio}})
        st.session_state.da_source_steps = src_steps

        if src_steps:
            st.markdown("Current source steps:")
            for s in src_steps:
                st.write(f"- {s['type']}: {s['params']}")

        st.markdown("---")

        # ----------------------------------------------------------------
        # Target Data Augmentation
        # ----------------------------------------------------------------
        st.markdown("#### Target Data Augmentation")
        st.session_state.da_target_pct = st.number_input(
            "Add % of augmented samples",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=st.session_state.da_target_pct,
            help="Fraction of new target samples to generate"
        )
        tgt_steps = []
        if st.checkbox("Magnitude Warping", key="tgt_mw"):
            sigma = st.number_input("  MW σ", min_value=0.0, value=1.0, key="tgt_mw_sigma")
            tgt_steps.append({"type": "MagnitudeWarping", "params": {"sigma": sigma}})
        if st.checkbox("Gaussian Noise", key="tgt_gn"):
            mean = st.number_input("  GN mean", value=0.0, key="tgt_gn_mean")
            std  = st.number_input("  GN std",  min_value=0.0, value=0.1, key="tgt_gn_std")
            tgt_steps.append({"type": "GaussianNoise", "params": {"mean": mean, "std": std}})
        if st.checkbox("Window Warping", key="tgt_ww"):
            scales_str = st.text_input("  WW scales (comma‑sep)", value="0.5", key="tgt_ww_scales")
            window_ratio = st.number_input("  WW window_ratio", min_value=0.0, max_value=1.0, value=0.5, key="tgt_ww_ratio")
            try:
                scales = [float(x.strip()) for x in scales_str.split(",") if x.strip()]
            except ValueError:
                st.error("WW scales must be numbers")
                scales = []
            tgt_steps.append({"type": "WindowWarping", "params": {"scales": scales, "window_ratio": window_ratio}})
        st.session_state.da_target_steps = tgt_steps

        if tgt_steps:
            st.markdown("Current target steps:")
            for s in tgt_steps:
                st.write(f"- {s['type']}: {s['params']}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # Validate & save
        # ----------------------------------------------------------------
        if st.button("Validate Data Augmentation", use_container_width=True):
            errs = []
            if st.session_state.da_source_pct < 0:
                errs.append("Source percentage must be > 0")
            if not st.session_state.da_source_steps:
                errs.append("Add at least one source step")
            if st.session_state.da_target_pct < 0:
                errs.append("Target percentage must be > 0")
            if not st.session_state.da_target_steps:
                errs.append("Add at least one target step")

            if errs:
                for e in errs:
                    st.error(e)
            else:
                cfg = {
                    "source_data_augmentation": {
                        "data_augmentation_options": [
                            {
                                "add_percentage_of_augmented_samples": st.session_state.da_source_pct,
                                "augmentation_steps": st.session_state.da_source_steps
                            }
                        ]
                    },
                    "target_data_augmentation": {
                        "data_augmentation_options": [
                            {
                                "add_percentage_of_augmented_samples": st.session_state.da_target_pct,
                                "augmentation_steps": st.session_state.da_target_steps
                            }
                        ]
                    }
                }
                st.session_state.data_augmentation = cfg
                st.session_state.data_augmentation_valid = True
                st.json(cfg, expanded=True)
                st.success("Data Augmentation is valid! You can proceed to next step.")


# ----------------------------------------------------------------------
# 8. Preprocessing
# ----------------------------------------------------------------------

def preprocessing():
    """
        Preprocessing
            - This function will be used to setup the data augmentation.
            - It will contain the following fields:
                - Normalization
                    - Standard Normalization
                    - No Normalization
    """

    with st.expander("Preprocessing Configuration", expanded=True):
        st.subheader("Preprocessing")
        
        # Normalization (Radio Buttons)
        normalization_options = ["Standard Normalization", "No Normalization"]
        st.session_state.normalization = st.radio(
            "Normalization",
            options=normalization_options,
            index=normalization_options.index(st.session_state.normalization)
                if st.session_state.normalization in normalization_options else 0,
            help="Select the normalization method."
        )
        
        # Validate the data augmentation inputs
        if st.button("Validate Preprocessing", use_container_width=True):
            # Check if each field is filled
            # Normalization
            if not st.session_state.normalization:
                st.error("Please select a Normalization method!")
            # If all fields are valid
            else:
                # Save the data augmentation to session state
                st.session_state.preprocessing = {
                    "normalization": st.session_state.normalization
                }
                st.session_state.preprocessing_valid = True
                st.success("Preprocessing is valid! You can proceed to the next step.")


# ----------------------------------------------------------------------
# 9. Backbone Model
# ----------------------------------------------------------------------

def backbone_model():
    """
        Backbone Model
            - This function will be used to setup the backbone model.
            - It will contain the following fields:
                - Backbone Model
                    - CNN
                    - RESNET18
                    - TCN
                - Hyperparameters
                    - Kernel Size
                    - Stride
                    - Dropout
                    
                - Only when the model is selected, the following fields will be shown.
                - Model Specific Parameters
                    - Common to all models:
                        - Final Outcome Channels
                        - Features Length
                    - CNN:
                        - Mid Channels
                    - RESNET18:
                        - None
                    - TCN:
                        - tcn_layers
                        - tcn_final_out_channels
                        - tcn_kernel_size
                        - tcn_dropout
    """

    with st.expander("Backbone Model Configuration", expanded=True):
        st.subheader("Backbone Model")
        
        # Backbone Model (Drop Down Menu)
        backbone_options = ["Select Model", "CNN", "RESNET18", "TCN"]#, "LTCN"]
        st.session_state.backbone_model = st.selectbox(
            "Backbone Model",
            options=backbone_options,
            index=backbone_options.index(st.session_state.backbone_model)
                    if st.session_state.backbone_model in backbone_options else 0,
            help="Select the backbone model."
        )
        
        # Hyperparameters
        st.session_state.kernel_size = st.number_input(
            "Kernel Size",
            min_value=1,
            value=st.session_state.kernel_size,
            help="Enter the kernel size."
        )
        st.session_state.stride = st.number_input(
            "Stride",
            min_value=1,
            value=st.session_state.stride,
            help="Enter the stride."
        )
        st.session_state.dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.dropout,
            format="%.2f",
            help="Enter the dropout rate."
        )

        # Common parameters for all backbone models
        st.session_state.final_outcome_channels = st.number_input(
            "Final Outcome Channels",
            min_value=1,
            value=st.session_state.final_outcome_channels,
            help="Enter the final outcome channels."
        )
        st.session_state.features_length = st.number_input(
            "Features Length",
            min_value=1,
            value=st.session_state.features_length,
            help="Enter the features length."
        )

        
        # Model Specific Parameters (only shown when a model other than "Select Model" is chosen)
        if st.session_state.backbone_model != "Select Model":
            
            
            # Model-specific parameters
            if st.session_state.backbone_model == "CNN":
                st.markdown("#### Model Specific Parameters")
                st.session_state.mid_channels = st.number_input(
                    "Mid Channels",
                    min_value=1,
                    value=st.session_state.mid_channels,
                    help="Enter the mid channels."
                )
            elif st.session_state.backbone_model == "LTCN":
                st.markdown("#### Model Specific Parameters")
                #st.session_state.ltcn_hidden_size = st.number_input(
                #    "LTCN Hidden Size",
                #    min_value=1,
                #    value=st.session_state.ltcn_hidden_size if "ltcn_hidden_size" in st.session_state else 64,
                #    help="Enter the LTCN hidden size."
                #)
                st.session_state.ltcn_hidden_size = 0
                st.session_state.ltcn_ode_solver_unfolds = st.number_input(
                    "LTCN ODE Solver Unfolds",
                    min_value=1,
                    value=st.session_state.ltcn_ode_solver_unfolds if "ltcn_ode_solver_unfolds" in st.session_state else 6,
                    help="Enter the LTCN ODE solver unfolds."
                )
            elif st.session_state.backbone_model == "TCN":
                st.markdown("#### Model Specific Parameters")
                st.session_state.tcn_layers = st.text_input(
                    "TCN Layers (comma separated)",
                    value=st.session_state.tcn_layers if "tcn_layers" in st.session_state else "75,150",
                    help="Enter TCN layers as comma separated integers."
                )
                st.session_state.tcn_kernel_size = st.number_input(
                    "TCN Kernel Size",
                    min_value=1,
                    value=st.session_state.tcn_kernel_size if "tcn_kernel_size" in st.session_state else 17,
                    help="Enter the TCN kernel size."
                )
                st.session_state.tcn_dropout = st.number_input(
                    "TCN Dropout",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.tcn_dropout if "tcn_dropout" in st.session_state else 0.0,
                    format="%.2f",
                    help="Enter the TCN dropout rate."
                )
                
        # Validate the backbone model inputs
        if st.button("Validate Backbone Model", use_container_width=True):
            # Check if each field is filled
            # Backbone Model
            if not st.session_state.backbone_model or st.session_state.backbone_model == "Select Model":
                st.error("Please select a Backbone Model!")
            # Kernel Size
            elif st.session_state.kernel_size < 1:
                st.error("Kernel Size must be at least 1!")
            # Stride
            elif st.session_state.stride < 1:
                st.error("Stride must be at least 1!")
            # Dropout
            elif not (0 <= st.session_state.dropout <= 1):
                st.error("Dropout must be between 0 and 1!")
            # If all fields are valid
            else:
                # Save the backbone model configuration to session state
                backbone_config = {
                    "backbone_model": st.session_state.backbone_model,
                    "kernel_size": st.session_state.kernel_size,
                    "stride": st.session_state.stride,
                    "dropout": st.session_state.dropout,
                    "final_outcome_channels": st.session_state.final_outcome_channels,
                    "features_length": st.session_state.features_length
                }
                # Append model-specific parameters
                if st.session_state.backbone_model == "CNN":
                    backbone_config["mid_channels"] = st.session_state.mid_channels
                elif st.session_state.backbone_model == "LTCN":
                    backbone_config["ltcn_hidden_size"] = st.session_state.ltcn_hidden_size
                    backbone_config["ltcn_ode_solver_unfolds"] = st.session_state.ltcn_ode_solver_unfolds
                elif st.session_state.backbone_model == "TCN":
                    backbone_config["tcn_layers"] = st.session_state.tcn_layers
                    backbone_config["tcn_kernel_size"] = st.session_state.tcn_kernel_size
                    backbone_config["tcn_dropout"] = st.session_state.tcn_dropout
                st.session_state.backbone_model_config = backbone_config
                st.json(st.session_state.backbone_model_config, expanded=True)
                st.session_state.backbone_model_valid = True
                st.success("Backbone Model is valid! You can proceed to the next step.")


# ----------------------------------------------------------------------
# 10. Training Strategy
# ----------------------------------------------------------------------

def training_strategy():
    """
        Training Strategy
            - This function will be used to setup the training strategy.
            - It will contain the following fields:
                - Algorithm
                    - NO_ADAPT
                    - Deep_Coral
                    - MMDA
                    - DANN
                    - CDAN
                    - DIRT
                    - DSAN
                    - HoMM
                    - CoDATS
                    - AdvSKM
                    - SASA
                    - CoTMix
                    - TARGET_ONLY
                    
                - Training Parameters
                    - Number of Epochs
                    - Batch Size
                    - Weight Decay
                    - Step Size (optional)
                    - Learning Rate Decay (optional)
                    
                - Only when the algorithm is selected, the following fields will be shown:
                    - Algorithm Specific Parameters (derived from predefined lists)
                    - Each Algorithm has its own set of parameters.
                    - These parameters will be shown only when the algorithm is selected.
    """

    # Create an expander for the Training Strategy configuration
    with st.expander("Training Strategy Configuration", expanded=True):
        st.subheader("Training Strategy")
        
        # --- Algorithm Selection ---
        algorithm_options = [
            "Select Algorithm",
            "NO_ADAPT",
            "Deep_Coral",
            "MMDA",
            "DANN",
            "CDAN",
            "DIRT",
            "DSAN",
            "HoMM",
            "CoDATS",
            "AdvSKM",
            "SASA",
            "CoTMix",
            "TARGET_ONLY"
        ]
        st.session_state.algorithm = st.selectbox(
            "Algorithm",
            options=algorithm_options,
            index=algorithm_options.index(st.session_state.algorithm) if st.session_state.algorithm in algorithm_options else 0,
            help="Select the training algorithm."
        )
        
        # --- Training Parameters ---
        # Check if hyperparameter tuning is enabled to conditionally show/hide parameters
        hyperparameter_tuning_enabled = (
            st.session_state.get("hyperparameter_tuning_config", {}).get("hyperparameter_tuning") == "Hyperparameter Tuning"
        )
        
        if hyperparameter_tuning_enabled:
            st.info("🔧 **Hyperparameter Tuning is enabled.** The following training parameters are configured in the Hyperparameter Configuration section and will be used during training/sweeping:")
            
            # Show read-only information about hyperparameter tuning configuration
            hparams_config = st.session_state.get("hyperparameter_tuning_config", {})
            if "num_epochs" in hparams_config:
                epochs_values = hparams_config.get("num_epochs", {}).get("values", ["N/A"])
                st.write(f"📊 **Number of Epochs**: {epochs_values}")
            if "batch_size" in hparams_config:
                batch_values = hparams_config.get("batch_size", {}).get("values", ["N/A"])
                st.write(f"📊 **Batch Size**: {batch_values}")
            if "learning_rate" in hparams_config:
                lr_values = hparams_config.get("learning_rate", {}).get("values", ["N/A"])
                st.write(f"📊 **Learning Rate**: {lr_values}")
            if "disc_lr" in hparams_config:
                disc_lr_values = hparams_config.get("disc_lr", {}).get("values", ["N/A"])
                st.write(f"📊 **Discriminator Learning Rate**: {disc_lr_values}")
            if "weight_decay" in hparams_config:
                wd_values = hparams_config.get("weight_decay", {}).get("values", ["N/A"])
                st.write(f"📊 **Weight Decay**: {wd_values}")
            if "step_size" in hparams_config:
                ss_values = hparams_config.get("step_size", {}).get("values", ["N/A"])
                st.write(f"📊 **Step Size**: {ss_values}")
            if "gamma" in hparams_config:
                gamma_values = hparams_config.get("gamma", {}).get("values", ["N/A"])
                st.write(f"📊 **Gamma**: {gamma_values}")
            if "optimizer" in hparams_config:
                opt_values = hparams_config.get("optimizer", {}).get("values", ["N/A"])
                st.write(f"📊 **Optimizer**: {opt_values}")
                
            # Still need these values for validation and fallback, but use defaults
            if "number_of_epochs" not in st.session_state:
                st.session_state.number_of_epochs = 10
            if "batch_size" not in st.session_state:
                st.session_state.batch_size = 32
            if "weight_decay" not in st.session_state:
                st.session_state.weight_decay = 0.0001
            if "step_size" not in st.session_state:
                st.session_state.step_size = 10
            if "lr_decay" not in st.session_state:
                st.session_state.lr_decay = 0.5
        else:
            st.session_state.number_of_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                value=st.session_state.number_of_epochs,
                help="Enter the number of epochs."
            )
            st.session_state.batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                value=st.session_state.batch_size,
                help="Enter the batch size."
            )
            st.session_state.weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weight_decay,
                format="%.4f",
                help="Enter the weight decay."
            )
            # Optional fields - Must be allowed to be selected as parameters or not
            st.session_state.step_size = st.number_input(
                "Step Size",
                min_value=1,
                value=st.session_state.step_size,
                help="Enter the step size."
            )
        
        # Learning Rate Decay is always shown since it's not part of hyperparameter tuning
        st.session_state.lr_decay = st.number_input(
            "Learning Rate Decay",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.lr_decay,
            format="%.4f",
            help="Enter the learning rate decay."
        )
        
        # --- Algorithm Specific Parameters ---
        # Reference lists for algorithm-specific keys (do not use the reference dicts here).
        # Any common parameter (weight_decay, step_size, lr_decay) will be overridden by the training parameters.
        common_keys = ["weight_decay", "step_size", "lr_decay"]
        algo_specific_keys = {
            "NO_ADAPT": ["learning_rate", "src_cls_loss_wt"],
            "TARGET_ONLY": ["learning_rate", "trg_cls_loss_wt"],
            "SASA": ["domain_loss_wt", "learning_rate", "src_cls_loss_wt", "weight_decay"],
            "DDC": ["learning_rate", "mmd_wt", "src_cls_loss_wt", "domain_loss_wt", "weight_decay"],
            "CoDATS": ["domain_loss_wt", "learning_rate", "src_cls_loss_wt", "weight_decay"],
            "DANN": ["domain_loss_wt", "learning_rate", "src_cls_loss_wt", "weight_decay", "step_size", "lr_decay"],
            "DIRT": ["cond_ent_wt", "domain_loss_wt", "learning_rate", "src_cls_loss_wt", "vat_loss_wt", "weight_decay"],
            "DSAN": ["learning_rate", "mmd_wt", "src_cls_loss_wt", "domain_loss_wt", "weight_decay"],
            "MMDA": ["cond_ent_wt", "coral_wt", "learning_rate", "mmd_wt", "src_cls_loss_wt", "weight_decay"],
            "Deep_Coral": ["coral_wt", "learning_rate", "src_cls_loss_wt", "weight_decay"],
            "CDAN": ["cond_ent_wt", "domain_loss_wt", "learning_rate", "src_cls_loss_wt", "weight_decay"],
            "AdvSKM": ["domain_loss_wt", "learning_rate", "src_cls_loss_wt", "weight_decay"],
            "HoMM": ["hommd_wt", "learning_rate", "src_cls_loss_wt", "domain_loss_wt", "weight_decay"],
            "CoTMix": ["learning_rate", "mix_ratio", "temporal_shift", "src_cls_weight", "src_supCon_weight", "trg_cont_weight", "trg_entropy_weight"],
            "MCD": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt"]
        }

        # --- Algorithm Specific Hyperparameters ---
        algo_specific_hparams_keys = {
            "DANN": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt"],
            "AdvSKM": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt"],
            "CoDATS": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt"],
            "CDAN": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt", "cond_ent_wt"],
            "Deep_Coral": ["learning_rate", "src_cls_loss_wt", "coral_wt"],
            "DIRT": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt", "cond_ent_wt", "vat_loss_wt"],
            "HoMM": ["learning_rate", "src_cls_loss_wt", "hommd_wt"],
            "MMDA": ["learning_rate", "src_cls_loss_wt", "coral_wt", "cond_ent_wt", "mmd_wt"],
            "DSAN": ["learning_rate", "src_cls_loss_wt", "mmd_wt"],
            "DDC": ["learning_rate", "src_cls_loss_wt", "mmd_wt"],
            "SASA": ["learning_rate", "src_cls_loss_wt", "domain_loss_wt"],
            "CoTMix": ["learning_rate", "temporal_shift", "src_cls_weight", "mix_ratio", "src_supCon_weight", "trg_cont_weight", "trg_entropy_weight"],
        }

        hyperparams_dict = {}
        
        # Check if a valid algorithm is selected to display algorithm-specific parameters.
        if st.session_state.algorithm != "Select Algorithm":
            # Retrieve the list of keys for the selected algorithm and remove any common training keys.
            selected_algo_keys = algo_specific_keys.get(st.session_state.algorithm, [])
            filtered_algo_keys = [key for key in selected_algo_keys if key not in common_keys]
            
            if st.session_state.get("hyperparameter_tuning_config", {}).get("hyperparameter_tuning") != "Hyperparameter Tuning":
                if filtered_algo_keys:
                    st.markdown("#### Algorithm Specific Parameters")
                    for param in filtered_algo_keys:
                        # Use an integer input for parameters containing "shift"; otherwise, use a float input.
                        if "shift" in param:
                            st.session_state[param] = st.number_input(
                                param.replace("_", " ").title(),
                                min_value=0,
                                value=st.session_state.get(param, 9),
                                step=1,
                                help=f"Enter the value for {param} (integer)."
                            )
                        else:
                            default_val = st.session_state.get(param, 0.9)

                            if isinstance(default_val, list):
                                if len(default_val) > 0:
                                    default_val = default_val[0]
                                else:
                                    default_val = 0.9

                            
                            st.session_state[param] = st.number_input(
                                param.replace("_", " ").title(),
                                min_value=0.0,
                                value=default_val,
                                format="%.4f",
                                help=f"Enter the value for {param} (float)."
                            )

            # --- Additional Discriminator Parameters for Discriminator-Based Algorithms ---
            if st.session_state.algorithm in ["DANN", "CDAN", "DIRT", "CoDATS", "AdvSKM"]:
                st.markdown("#### Discriminator Parameters")
                for param, default_val in zip(
                    ["disc_hid_dim", "hidden_dim", "DSKN_disc_hid"],
                    [st.session_state.get("disc_hid_dim", 64),
                     st.session_state.get("hidden_dim", 500),
                     st.session_state.get("DSKN_disc_hid", 128)]
                ):
                    st.session_state[param] = st.number_input(
                        param.replace("_", " ").title(),
                        min_value=1,
                        value=default_val,
                        step=1,
                        help=f"Enter the value for {param} (integer)."
                    )

            if st.session_state.get("hyperparameter_tuning_config", {}).get("hyperparameter_tuning") == "Hyperparameter Tuning":
                st.markdown("#### Algorithm Specific Hyperparameters")
                # Check if the selected algorithm has hyperparameters defined in the sweep dictionary.
                if st.session_state.algorithm in sweep_alg_hparams:
                    algo_hparams = sweep_alg_hparams.get(st.session_state.algorithm, {})
                    
                    # Loop through each hyperparameter and generate input fields based on its configuration.
                    for param, cfg in algo_hparams.items():
                        param_label = param.replace('_', ' ').title()
                        
                        if "values" in cfg:
                            # Special handling for parameters like learning_rate that must accept multiple values.
                            if param == "learning_rate":
                                # # Convert the default values from the config into strings.
                                # default_values = [str(v) for v in cfg["values"]]
                                
                                # # Provide a multiselect to pick from the predefined set.
                                # selected_values = st.multiselect(
                                #     label=f"{param_label} (Select one or more)",
                                #     options=default_values,
                                #     default=default_values,
                                #     key=f"{st.session_state.algorithm}_{param}_multiselect"
                                # )
                                
                                # # Provide a text input for additional values.
                                # additional_str = st.text_input(
                                #     "Add additional values (comma separated)",
                                #     key=f"{st.session_state.algorithm}_{param}_additional"
                                # )
                                
                                # # Process any additional values, trimming extra whitespace.
                                # additional_values = [x.strip() for x in additional_str.split(",") if x.strip()]
                                
                                # # Combine both the selected default values and the additional values.
                                # # Convert all values to float if possible.
                                # try:
                                #     combined_values = [float(x) for x in (selected_values + additional_values)]
                                # except ValueError:
                                #     st.error(f"One or more entered learning rate values for {param_label} are not valid numbers.")
                                #     combined_values = []
                                
                                # hyperparams_dict[param] = {"values": combined_values}
                                pass

                            else:
                                # For parameters with fixed values (other than learning rate), use a selectbox.
                                chosen_value = st.selectbox(
                                    label=param_label,
                                    options=cfg["values"],
                                    key=f"{st.session_state.algorithm}_{param}"
                                )
                                st.session_state[param] = chosen_value
                                
                                hyperparams_dict[param] = {"values": [chosen_value]}
                        
                        elif "distribution" in cfg:
                            # For distribution-based hyperparameters,
                            # create a row with four columns: label, distribution drop-down, min, and max.
                            cols = st.columns(4)
                            
                            # Column 1: show the parameter label.
                            cols[0].markdown(f"**{param_label}**")
                            
                            # Column 2: a selectbox for the distribution type.
                            distribution_options = ["uniform"]
                            default_dist = cfg.get("distribution", "uniform")
                            default_index = distribution_options.index(default_dist) if default_dist in distribution_options else 0
                            selected_dist = cols[1].selectbox(
                                label="Distribution",
                                options=distribution_options,
                                index=default_index,
                                key=f"{st.session_state.algorithm}_{param}_dist"
                            )
                            
                            # Column 3: number input for the minimum value.
                            min_val = cols[2].number_input(
                                label="Min",
                                min_value=0.0,
                                value=float(cfg.get("min", 0.0)),
                                format="%.4f",
                                key=f"{st.session_state.algorithm}_{param}_min"
                            )
                            
                            # Column 4: number input for the maximum value.
                            max_val = cols[3].number_input(
                                label="Max",
                                min_value=0.0,
                                value=float(cfg.get("max", 1.0)),
                                format="%.4f",
                                key=f"{st.session_state.algorithm}_{param}_max"
                            )

                            hyperparams_dict[param] = {
                                "distribution": selected_dist,
                                "min": min_val,
                                "max": max_val,
                            }
                
                else:
                    st.info("No sweep hyperparameters defined for this algorithm.")

        
        # Validate the training strategy inputs
        if st.button("Validate Training Strategy", use_container_width=True):
            try:
                # Check if each field is filled with safe access
                algorithm = st.session_state.get("algorithm", "")
                number_of_epochs = st.session_state.get("number_of_epochs", 0)
                batch_size = st.session_state.get("batch_size", 0)
                weight_decay = st.session_state.get("weight_decay", 0.0)
                step_size = st.session_state.get("step_size", 1)
                lr_decay = st.session_state.get("lr_decay", 0.5)
                
                # Algorithm
                if not algorithm or algorithm == "Select Algorithm":
                    st.error("Please select an Algorithm!")
                # Number of Epochs
                elif number_of_epochs < 1:
                    st.error("Number of Epochs must be at least 1!")
                # Batch Size
                elif batch_size < 1:
                    st.error("Batch Size must be at least 1!")
                # Weight Decay
                elif not (0 <= weight_decay <= 1):
                    st.error("Weight Decay must be between 0 and 1!")
                # If all fields are valid
                else:
                    alg_params = {}
                    if algorithm != "Select Algorithm":
                        # Safely get algorithm-specific parameters
                        selected_algo_keys = algo_specific_keys.get(algorithm, [])
                        for key in selected_algo_keys:
                            if key not in common_keys:
                                param_value = st.session_state.get(key)
                                if param_value is not None:
                                    alg_params[key] = param_value
                        
                        # Include additional discriminator parameters if applicable.
                        if algorithm in ["DANN", "CDAN", "DIRT", "CoDATS", "AdvSKM"]:
                            for key in ["disc_hid_dim", "hidden_dim", "DSKN_disc_hid"]:
                                param_value = st.session_state.get(key)
                                if param_value is not None:
                                    alg_params[key] = param_value

                    training_strategy_config = {
                        "algorithm": algorithm,
                        "num_epochs": number_of_epochs,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "step_size": step_size,
                        "lr_decay": lr_decay,
                        **alg_params,
                    }

                    # Add hyperparameters if hyperparameter tuning is enabled
                    if st.session_state.get("hyperparameter_tuning_config", {}).get("hyperparameter_tuning") == "Hyperparameter Tuning":
                        if hyperparams_dict:
                            training_strategy_config[algorithm] = hyperparams_dict
                    
                    st.json(training_strategy_config, expanded=True)
                    st.session_state.training_strategy = training_strategy_config
                    st.session_state.training_strategy_valid = True
                    st.success("Training Strategy is valid! You can proceed to the next step.")
                    
            except Exception as e:
                st.error(f"An error occurred during validation: {e}")
                st.error("Please check your input values and try again.")


# ----------------------------------------------------------------------
# 11. Results
# ----------------------------------------------------------------------

def display_metrics(metrics: dict):
    # Show each metric in a column
    cols = st.columns(3)
    metric_names = ["acc", "f1_score", "auroc"]
    for name, col in zip(metric_names, cols):
        val = metrics.get(name, 0)
        if 0 <= val <= 1:
            col.metric(label=name.upper(), value=f"{val:.2%}")
        else:
            col.metric(label=name.upper(), value=f"{val}")


def results(summary_best):
    """
        Results
    """
    st.write("### Best Results:")
    if st.session_state.results:
        display_metrics(st.session_state.results)
    else:
        st.info("No results dictionary found, but Step 4 was validated. Please re-run Step 4.")
        
    st.markdown("---")
    st.subheader("Detailed Results")

    try:
        # Safe access to session state for building logs folder path
        evaluation_setup = st.session_state.get('evaluation_setup', {})
        training_strategy = st.session_state.get('training_strategy', {})
        
        dataset_choice = evaluation_setup.get('dataset_choice', 'HHAR')
        algorithm = training_strategy.get('algorithm', 'DANN')
        experiment_name = evaluation_setup.get('experiment_name', 'default_experiment')
        
        # Build the logs folder path
        logs_folder_path = f"ADATime/experiments_logs/{dataset_choice}/{algorithm}_{experiment_name}"

        absolute_path = os.path.abspath(logs_folder_path)

        if os.name == 'nt':  # Windows
            file_url = f"file:///{absolute_path.replace(os.sep, '/')}"
        else:  # macOS/Linux
            file_url = f"file://{absolute_path}"

        # When clicked open the logs folder
        # st.markdown("##### Logs Folder")
        # st.markdown(f"[Open Logs Folder]({file_url})")
        # st.markdown("---")

        # Display the best_results.csv inside the logs folder
        st.markdown("##### Best Results")
        df_best = pd.read_csv(logs_folder_path + "/best_results.csv")
        st.dataframe(df_best, use_container_width=True)
        st.markdown("---")

        # Display the last_results.csv inside the logs folder
        st.markdown("##### Last Results")
        df_last = pd.read_csv(logs_folder_path + "/last_results.csv")
        st.dataframe(df_last, use_container_width=True)
        st.markdown("---")

        # Display the results.csv
        st.markdown("##### Results")
        df_results = pd.read_csv(logs_folder_path + "/results.csv")
        st.dataframe(df_results, use_container_width=True)
        st.markdown("---")

        # Display the risks.csv
        st.markdown("##### Risks")
        df_risks = pd.read_csv(logs_folder_path + "/risks.csv")
        st.dataframe(df_risks, use_container_width=True)
        
    except KeyError as e:
        st.error(f"Missing required configuration: {e}")
    except FileNotFoundError as e:
        st.error(f"Results files not found: {e}")
    except Exception as e:
        st.error(f"An error occurred while displaying results: {e}")


# -----------------------------------------------------------------------
# 12. Run Experiment
# -----------------------------------------------------------------------

def run_experiment_ADATime():
    """
        Run the experiment
            - This function will be used to run the experiment.
            - It will contain the following fields:
                - Start Training
                - Stop Training
                - Show Logs
    """
    # Check if required configurations are available
    required_configs = ['evaluation_setup', 'training_strategy', 'backbone_model_config', 'preprocessing']
    missing_configs = [config for config in required_configs if config not in st.session_state]
    
    if missing_configs:
        st.error(f"Missing required configurations: {', '.join(missing_configs)}")
        return None

    # Additional check for data_augmentation which is used in the function
    if 'data_augmentation' not in st.session_state:
        st.error("Missing data_augmentation configuration")
        return None

    # Determine the available device (MPS > CUDA > CPU)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Safe access to session state with proper error handling
        evaluation_setup = st.session_state.get('evaluation_setup', {})
        training_strategy = st.session_state.get('training_strategy', {})
        backbone_model_config = st.session_state.get('backbone_model_config', {})
        preprocessing = st.session_state.get('preprocessing', {})
        data_augmentation = st.session_state.get('data_augmentation', {})

        args_for_train = argparse.Namespace(
            phase='train',
            save_dir='ADATime/experiments_logs',
            exp_name=evaluation_setup.get("experiment_name", "default_experiment"),
            da_method=training_strategy.get("algorithm", "DANN"),
            data_path="ADATIME_data",
            dataset=evaluation_setup.get("dataset_choice", "HHAR"),
            backbone=backbone_model_config.get("backbone_model", "CNN"),
            num_runs=evaluation_setup.get("number_of_runs", 1),
            device=device,
        )

        args_for_test = argparse.Namespace(
            phase='test',
            save_dir='ADATime/experiments_logs',
            exp_name=evaluation_setup.get("experiment_name", "default_experiment"),
            da_method=training_strategy.get("algorithm", "DANN"),
            data_path="ADATIME_data",
            dataset=evaluation_setup.get("dataset_choice", "HHAR"),
            backbone=backbone_model_config.get("backbone_model", "CNN"),
            num_runs=evaluation_setup.get("number_of_runs", 1),
            device=device,
        )

        dataset_class = get_dataset_class(evaluation_setup.get("dataset_choice", "HHAR"))
        dataset_config = dataset_class()

        # Common dataset configuration parameters
        dataset_config.scenarios = evaluation_setup.get("scenarios", [])
        dataset_config.stride = backbone_model_config.get("stride", 1)
        dataset_config.kernel_size = backbone_model_config.get("kernel_size", 3)
        dataset_config.final_out_channels = backbone_model_config.get("final_outcome_channels", 128)
        dataset_config.features_len = backbone_model_config.get("features_length", 128)
        dataset_config.normalize = True if preprocessing.get("normalization") == "Standard Normalization" else False
        dataset_config.data_augmentation_configs = data_augmentation

        # For Discriminator-based algorithms, set the discriminator parameters
        algorithm = training_strategy.get("algorithm", "")
        if algorithm in ["DANN", "CDAN", "DIRT", "CoDATS", "AdvSKM"]:
            dataset_config.disc_hid_dim = training_strategy.get("disc_hid_dim", 64)
            dataset_config.hidden_dim = training_strategy.get("hidden_dim", 64)
            dataset_config.DSKN_disc_hid = training_strategy.get("DSKN_disc_hid", 64)

        # Model-specific dataset configuration updates
        backbone = backbone_model_config.get("backbone_model", "CNN")
        if backbone == "CNN":
            # For CNN, update mid_channels if available
            if "mid_channels" in backbone_model_config:
                dataset_config.mid_channels = backbone_model_config["mid_channels"]
        elif backbone == "LTCN":
            # For LTCN, update hidden_size and ode_solver_unfolds
            dataset_config.hidden_size = backbone_model_config.get("ltcn_hidden_size", 64)
            dataset_config.ode_solver_unfolds = backbone_model_config.get("ltcn_ode_solver_unfolds", 6)
        elif backbone == "TCN":
            # For TCN, convert tcn_layers from string to list of integers and update TCN parameters
            tcn_layers_str = backbone_model_config.get("tcn_layers", "75,150")
            try:
                tcn_layers_list = [int(x.strip()) for x in tcn_layers_str.split(",")]
            except Exception as e:
                tcn_layers_list = [75, 150]
            dataset_config.tcn_layers = tcn_layers_list
            dataset_config.tcn_final_out_channels = backbone_model_config.get("tcn_final_out_channels", 32)
            dataset_config.tcn_kernel_size = backbone_model_config.get("tcn_kernel_size", 7)
            dataset_config.tcn_dropout = backbone_model_config.get("tcn_dropout", 0.2)

        # Determine which hyperparameters to use based on hyperparameter tuning setting
        hyperparameter_tuning_config = st.session_state.get("hyperparameter_tuning_config", {})
        if hyperparameter_tuning_config.get("hyperparameter_tuning") == "Hyperparameter Tuning":
            # Use hyperparameter tuning configuration - for normal training, pick first values from ranges
            # (For sweeps, all values will be explored automatically)
            training_hparams = {
                "algorithm": training_strategy.get("algorithm", "DANN"),
                "num_epochs": hyperparameter_tuning_config.get("num_epochs", {}).get("values", [100])[0],
                "batch_size": hyperparameter_tuning_config.get("batch_size", {}).get("values", [32])[0],
                "learning_rate": hyperparameter_tuning_config.get("learning_rate", {}).get("values", [0.001])[0],
                "disc_lr": hyperparameter_tuning_config.get("disc_lr", {}).get("values", [0.001])[0],
                "weight_decay": hyperparameter_tuning_config.get("weight_decay", {}).get("values", [1e-4])[0],
                "step_size": hyperparameter_tuning_config.get("step_size", {}).get("values", [50])[0],
                "gamma": hyperparameter_tuning_config.get("gamma", {}).get("values", [0.5])[0],
                "optimizer": hyperparameter_tuning_config.get("optimizer", {}).get("values", ["Adam"])[0],
                "lr_decay": training_strategy.get("lr_decay", 0.5),
            }
            
            # Add algorithm-specific parameters from training strategy
            for key, value in training_strategy.items():
                if key not in training_hparams and key != "algorithm":
                    training_hparams[key] = value
            
            st.info(f"🔧 Using Hyperparameter Tuning Configuration: num_epochs={training_hparams['num_epochs']}, batch_size={training_hparams['batch_size']}, learning_rate={training_hparams['learning_rate']}, weight_decay={training_hparams['weight_decay']}")
        else:
            # Use training strategy configuration (normal mode)
            training_hparams = training_strategy
            num_epochs = training_hparams.get('num_epochs', 'N/A')
            batch_size = training_hparams.get('batch_size', 'N/A')
            st.info(f"📝 Using Training Strategy Configuration: num_epochs={num_epochs}, batch_size={batch_size}")

        # Run training if phase is train
        if args_for_train.phase == 'train':
            trainer = TrainTrainer(args_for_train)
            trainer.fit(hparams=training_hparams, dataset_configs=dataset_config)

        # Run testing if phase is test
        if args_for_test.phase == 'test':
            trainer = TrainTrainer(args_for_test)
            summary_best = trainer.test(hparams=training_hparams, dataset_configs=dataset_config)
            return summary_best

        return None

    except KeyError as e:
        st.error(f"Missing required configuration key: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during experiment execution: {e}")
        return None
    

# ----------------------------------------------------------------------
# 13. Run Sweep
# ----------------------------------------------------------------------

def run_sweep_ADATime():
    """
        Run the sweep
    """

    with st.expander("Sweep", expanded=True):
        st.subheader("Run Sweep")

        # Check if required configurations are available
        required_configs = ['training_strategy', 'evaluation_setup', 'backbone_model_config', 
                          'hyperparameter_tuning_config', 'preprocessing', 'data_augmentation']
        missing_configs = [config for config in required_configs if config not in st.session_state]
        
        if missing_configs:
            st.error(f"Missing required configurations: {', '.join(missing_configs)}")
            return

        api = wandb.Api()
        try:
            username = api.default_entity
            st.write(f"WandB - Logged in as: {username}.")
        except Exception as e:
            st.error(f"WandB authentication failed: {e}")
            return

        st.markdown("<br>", unsafe_allow_html=True)

        st.success("Sweep started...")

        st.markdown("---")

        try:
            # Safe access to session state with proper error handling
            training_strategy = st.session_state.get('training_strategy', {})
            evaluation_setup = st.session_state.get('evaluation_setup', {})
            backbone_model_config = st.session_state.get('backbone_model_config', {})
            hyperparameter_tuning_config = st.session_state.get('hyperparameter_tuning_config', {})
            preprocessing = st.session_state.get('preprocessing', {})
            data_augmentation = st.session_state.get('data_augmentation', {})

            # Determine the available device (MPS > CUDA > CPU)
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

            args_for_sweep = argparse.Namespace(
                da_method=training_strategy.get("algorithm", "DANN"),
                data_path="ADATIME_data",
                dataset=evaluation_setup.get("dataset_choice", "HHAR"),
                backbone=backbone_model_config.get("backbone_model", "CNN"),
                num_runs=evaluation_setup.get("number_of_runs", 1),
                device=device,
                exp_name=hyperparameter_tuning_config.get("sweep_project_name", "ADATime_Sweep"),
                num_sweeps=hyperparameter_tuning_config.get("num_sweeps", 1),
                sweep_project_wandb=hyperparameter_tuning_config.get("sweep_project_wandb", "ADATime_Sweep"),
                wandb_entity="",
                hp_search_strategy="random",
                metric_to_minimize=hyperparameter_tuning_config.get("target_risk_score", "dev_risk"),
                save_dir='ADATime/experiments_logs/sweep_logs',
            )

            dataset_class = get_dataset_class(evaluation_setup.get("dataset_choice", "HHAR"))
            dataset_config = dataset_class()

            # Common dataset configuration parameters
            dataset_config.scenarios = evaluation_setup.get("scenarios", [])
            dataset_config.stride = backbone_model_config.get("stride", 1)
            dataset_config.kernel_size = backbone_model_config.get("kernel_size", 3)
            dataset_config.final_out_channels = backbone_model_config.get("final_outcome_channels", 128)
            dataset_config.features_len = backbone_model_config.get("features_length", 128)
            dataset_config.normalize = True if preprocessing.get("normalization") == "Standard Normalization" else False
            dataset_config.data_augmentation_configs = data_augmentation

            # For Discriminator-based algorithms, set the discriminator parameters
            algorithm = training_strategy.get("algorithm", "")
            if algorithm in ["DANN", "CDAN", "DIRT", "CoDATS", "AdvSKM"]:
                dataset_config.disc_hid_dim = training_strategy.get("disc_hid_dim", 64)
                dataset_config.hidden_dim = training_strategy.get("hidden_dim", 64)
                dataset_config.DSKN_disc_hid = training_strategy.get("DSKN_disc_hid", 64)

            # Model-specific dataset configuration updates
            backbone = backbone_model_config.get("backbone_model", "CNN")
            if backbone == "CNN":
                if "mid_channels" in backbone_model_config:
                    dataset_config.mid_channels = backbone_model_config["mid_channels"]
            elif backbone == "LTCN":
                dataset_config.hidden_size = backbone_model_config.get("ltcn_hidden_size", 64)
                dataset_config.ode_solver_unfolds = backbone_model_config.get("ltcn_ode_solver_unfolds", 6)
            elif backbone == "TCN":
                tcn_layers_str = backbone_model_config.get("tcn_layers", "75,150")
                try:
                    tcn_layers_list = [int(x.strip()) for x in tcn_layers_str.split(",")]
                except Exception as e:
                    tcn_layers_list = [75, 150]
                dataset_config.tcn_layers = tcn_layers_list
                dataset_config.tcn_final_out_channels = backbone_model_config.get("tcn_final_out_channels", 128)
                dataset_config.tcn_kernel_size = backbone_model_config.get("tcn_kernel_size", 17)
                dataset_config.tcn_dropout = backbone_model_config.get("tcn_dropout", 0.0)

            # Prepare UI hyperparameters for sweep (only when hyperparameter tuning is enabled)
            ui_sweep_hparams = None
            if hyperparameter_tuning_config.get("hyperparameter_tuning") == "Hyperparameter Tuning":
                ui_sweep_hparams = {
                    "num_epochs": hyperparameter_tuning_config.get("num_epochs", {"values": [100]}),
                    "batch_size": hyperparameter_tuning_config.get("batch_size", {"values": [32]}),
                    "learning_rate": hyperparameter_tuning_config.get("learning_rate", {"values": [0.001]}),
                    "disc_lr": hyperparameter_tuning_config.get("disc_lr", {"values": [0.001]}),
                    "weight_decay": hyperparameter_tuning_config.get("weight_decay", {"values": [1e-4]}),
                    "step_size": hyperparameter_tuning_config.get("step_size", {"values": [50]}),
                    "gamma": hyperparameter_tuning_config.get("gamma", {"values": [0.5]}),
                    "optimizer": hyperparameter_tuning_config.get("optimizer", {"values": ["Adam"]})
                }

            # Get combined sweep hyperparameters (training + algorithm-specific)
            combined_sweep_hparams = get_combined_sweep_hparams(
                ui_hparams=ui_sweep_hparams,
                algorithm=training_strategy.get("algorithm", "DANN")
            )

            sweep_hparams = {
                training_strategy.get("algorithm", "DANN"): combined_sweep_hparams
            }

            trainer = SweepTrainer(args_for_sweep)

            try:
                results_list = trainer.sweep(dataset_configs=dataset_config, sweep_hparams=sweep_hparams, hparams=training_strategy)
                results_df = pd.DataFrame(results_list)

                st.subheader("Detailed Sweep Results:")
                st.dataframe(results_df)
            except Exception as e:
                st.error(f"Sweep execution failed: {e}")
                return

            st.markdown("---")

            # Display the link to the sweep results WandB page
            st.subheader("WandB Sweep Results")
            sweep_project_wandb = hyperparameter_tuning_config.get("sweep_project_wandb", "ADATime_Sweep")
            wandb_url = f"https://wandb.ai/{username}/{sweep_project_wandb}/sweeps"
            st.markdown(f"[Open Sweep Results in WandB]({wandb_url})")

            st.markdown("---")

            st.markdown("<br>", unsafe_allow_html=True)

            st.success("Sweep finished...")

        except KeyError as e:
            st.error(f"Missing required configuration key: {e}")
            return
        except Exception as e:
            st.error(f"An error occurred during sweep execution: {e}")
            return



# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

# Load CSS
# load_css("ADATime/app/style.css")

# Header
header()

    
st.markdown("<br>", unsafe_allow_html=True)
# Evaluation Setup
evaluation_setup()

    # If Evaluation Setup is validated, proceed to Hyperparameter Tuning
if st.session_state.get("evaluation_setup_valid", False):
        st.markdown("<br>", unsafe_allow_html=True)
        hyperparameter_tuning()
        
        # If Hyperparameter Tuning is validated, proceed to Data Augmentation
        if st.session_state.get("hyperparameter_tuning_valid", False):
            st.markdown("<br>", unsafe_allow_html=True)
            data_augmentation()

            # If Data Augmentation is validated, proceed to Preprocessing
            if st.session_state.get("data_augmentation_valid", False):
                st.markdown("<br>", unsafe_allow_html=True)
                preprocessing()
            
                # If Preprocessing is validated, proceed to Backbone Model
                if st.session_state.get("preprocessing_valid", False):
                    st.markdown("<br>", unsafe_allow_html=True)
                    backbone_model()
                    
                    # If Backbone Model is validated, proceed to Training Strategy
                    if st.session_state.get("backbone_model_valid", False):
                        st.markdown("<br>", unsafe_allow_html=True)
                        training_strategy()
                        
                        # If Training Strategy is validated, allow running the experiment
                        if st.session_state.get("training_strategy_valid", False):
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.write("### Training and Evaluation")
                            st.write("Click the button below to start the training and evaluation process.")
                            st.markdown("<br>", unsafe_allow_html=True)#

                            if st.session_state.get("hyperparameter_tuning_config", {}).get('hyperparameter_tuning') != "Hyperparameter Tuning":
                                run_experiment = st.button("Run Experiment", use_container_width=True)

                                if run_experiment:
                                    with st.expander("Results", expanded=True):
                                        st.success("Running Experiment...")
                                        st.markdown("---")
                                        st.markdown("<br>", unsafe_allow_html=True)

                                        summary_best = run_experiment_ADATime()
                                        st.session_state.results = summary_best
                                        results(summary_best)

                                        st.markdown("---")
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        st.success("Finished Experiment...")
                            else:
                                run_sweep = st.button("Run Sweep", use_container_width=True)

                                st.markdown("<br>", unsafe_allow_html=True)

                                if run_sweep:
                                    run_sweep_ADATime()
