import os
import json
import time
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor

# === CONFIGURATION ===
BASE_PATH = "/home/aegis/ether/Research_HEP/Dataset_ver3/MC/reduce_root"
OUTPUT_DIR = "Regions_ver2"
LUMI = 36100.0  
REDUCTION = 0.6 
MAX_WORKERS = 3
BATCH_SIZE = "500 MB" # Controls memory usage per step

# Load Metadata
with open("mc_metadata.json", "r") as f:
    METADATA = json.load(f)

# Define Branches
BRANCHES = [
    # --- Global Event & MET ---
    "MET_Core_AnalysisMETAuxDyn_mpx",
    "MET_Core_AnalysisMETAuxDyn_mpy",
    "MET_Core_AnalysisMETAuxDyn_sumet",
    "EventInfoAuxDyn_mcEventWeights",
    
    # --- Small-R Jets (Selection & Cleaning) ---
    "AnalysisJetsAuxDyn_pt",
    "AnalysisJetsAuxDyn_eta",
    "AnalysisJetsAuxDyn_phi",
    "AnalysisJetsAuxDyn_NNJvtPass",
    
    # --- Large-R Jets (AD Features - Expanded) ---
    "AnalysisLargeRJetsAuxDyn_pt",
    "AnalysisLargeRJetsAuxDyn_eta",
    "AnalysisLargeRJetsAuxDyn_phi",
    "AnalysisLargeRJetsAuxDyn_m",
    "AnalysisLargeRJetsAuxDyn_Tau1_wta",
    "AnalysisLargeRJetsAuxDyn_Tau2_wta",
    "AnalysisLargeRJetsAuxDyn_Tau3_wta",
    
    # --- Lepton & Tau Vetoes ---
    "AnalysisElectronsAuxDyn_DFCommonElectronsLHTight",
    "AnalysisMuonsAuxDyn_muonType",
    "AnalysisMuonsAuxDyn_quality",
    "AnalysisTauJetsAuxDyn_JetDeepSetTight",
    
    # --- Flavor Tagging ---
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu",
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"
]


# === 1. CUTFLOW TRACKER CLASS ===
class CutflowTracker:
    def __init__(self):
        self.steps = []
        self.raw_counts = {}
        self.weighted_counts = {}

    def update(self, step_name, events):
        """Records the number of events and sum of weights at this step."""
        if step_name not in self.steps:
            self.steps.append(step_name)
        
        n_raw = len(events)
        w_sum = ak.sum(events["weight_phys"]) if len(events) > 0 else 0.0
        
        self.raw_counts[step_name] = self.raw_counts.get(step_name, 0) + n_raw
        self.weighted_counts[step_name] = self.weighted_counts.get(step_name, 0.0) + w_sum

    def save_csv(self, process_name, output_dir):
        """Saves the cutflow to a CSV file."""
        data = []
        # Handle case where steps might be empty if no events processed
        if not self.steps:
            return

        initial_w = self.weighted_counts.get(self.steps[0], 1.0)
        # Prevent divide by zero in efficiency calc if initial weight is 0
        if initial_w == 0: initial_w = 1.0 
        
        prev_w = initial_w

        for step in self.steps:
            raw = self.raw_counts[step]
            weighted = self.weighted_counts[step]
            
            # Efficiencies
            abs_eff = (weighted / initial_w) * 100
            rel_eff = (weighted / prev_w) * 100 if prev_w > 0 else 0
            
            data.append({
                "Step": step,
                "Raw Events": raw,
                "Weighted Yield": weighted,
                "Absolute Eff (%)": f"{abs_eff:.2f}",
                "Relative Eff (%)": f"{rel_eff:.2f}"
            })
            prev_w = weighted
            
        df = pd.DataFrame(data)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"cutflow_{process_name}.csv"), index=False)
        print(f"Saved cutflow for {process_name}")

# === 2. EVENT CLEANING (PRESELECTION) ===
def apply_preselection(events, tracker):
    """
    Applies cuts and updates the tracker. Returns filtered events.
    """
    if len(events) == 0: return events

    tracker.update("Initial", events)

    # --- Cut 1: Jet Kinematics ---
    jet_pt = events["AnalysisJetsAuxDyn_pt"] / 1000.0
    jet_eta = events["AnalysisJetsAuxDyn_eta"]
    
    # Acceptance & Counts
    in_acceptance = abs(jet_eta) < 2.8
    accepted_pts = jet_pt[in_acceptance]
    has_two_jets = ak.num(accepted_pts, axis=1) >= 2
    
    # Leading/Subleading Pt
    leading_pt = ak.pad_none(accepted_pts, 2, axis=1)[:, 0]
    subleading_pt = ak.pad_none(accepted_pts, 2, axis=1)[:, 1]
    
    # Fill None with False to safely handle empty arrays
    pass_jet_kin = (
        (ak.fill_none(leading_pt > 250, False)) & 
        (ak.fill_none(subleading_pt > 30, False)) & 
        has_two_jets
    )
    
    events = events[pass_jet_kin]
    tracker.update("Jet Selection", events)
    if len(events) == 0: return events

    # --- Cut 2: Large-R Jets ---
    ljet_pt = events["AnalysisLargeRJetsAuxDyn_pt"] / 1000.0
    pass_large_jet = ak.num(ljet_pt, axis=1) >= 2
    
    events = events[pass_large_jet]
    tracker.update("Large-R Jet >= 2", events)
    if len(events) == 0: return events

    # --- Cut 3: JVT ---
    # Re-access sliced branches
    jvt_pass = events["AnalysisJetsAuxDyn_NNJvtPass"]
    jet_pt = events["AnalysisJetsAuxDyn_pt"] / 1000.0
    jet_eta = events["AnalysisJetsAuxDyn_eta"]
    
    low_pt_mask = (jet_pt < 60) & (abs(jet_eta) < 2.4)
    # If low_pt, must pass JVT. Else (high pt), True.
    pass_jvt = ak.all(ak.where(low_pt_mask, jvt_pass, True), axis=1)
    
    events = events[pass_jvt]
    tracker.update("JVT Cleaning", events)
    if len(events) == 0: return events

    # --- Cut 4: MET & dPhi Calculation ---
    met_px = events["MET_Core_AnalysisMETAuxDyn_mpx"][:, 0] / 1000.0
    met_py = events["MET_Core_AnalysisMETAuxDyn_mpy"][:, 0] / 1000.0
    met = events["MET_Core_AnalysisMETAuxDyn_sumet"][:, 0] / 1000.0
    
    # Simple Vector Sum of Jets
    jet_phi = events["AnalysisJetsAuxDyn_phi"]
    jet_px = (events["AnalysisJetsAuxDyn_pt"] / 1000.0) * np.cos(jet_phi)
    jet_py = (events["AnalysisJetsAuxDyn_pt"] / 1000.0) * np.sin(jet_phi)
    
    sum_j_px = ak.sum(jet_px, axis=1)
    sum_j_py = ak.sum(jet_py, axis=1)
    
    # Recalculate MET
    met_recalc_px = -(sum_j_px + met_px)
    met_recalc_py = -(sum_j_py + met_py)
    met_val = met
    met_phi_recalc = np.arctan2(met_recalc_py, met_recalc_px)
    
    # Save calculated MET for later usage
    events["met_recalc_pt"] = met_val
    events["met_recalc_phi"] = met_phi_recalc

    # # MET Cut
    # pass_met = met_val > 250
    # events = events[pass_met]
    # tracker.update("MET > 250", events)
    # if len(events) == 0: return events

    # dPhi Cut
    dphi = np.abs(events["AnalysisJetsAuxDyn_phi"] - events["met_recalc_phi"])
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    pass_dphi = ak.any(dphi < 2.0, axis=1)
    
    events = events[pass_dphi]
    tracker.update("dPhi(Jet, MET) < 2.0", events)

    # --- Cut 5: B-Tagging Prep ---
    pb = events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"]
    pc = events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc"]
    pu = events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu"]
    fc = 0.080
    
    # FIX: Handle RuntimeWarning for log(0)
    # If pb is 0, log is -inf. Subsequent > 1.42 check handles -inf correctly (False).
    # We suppress the warning to keep output clean.
    with np.errstate(divide='ignore'):
        dl1_score = np.log(pb / (fc * pc + (1 - fc) * pu + 1e-10))
    
    events["is_bjet"] = dl1_score > 1.45  
    events["n_bjets"] = ak.sum(events["is_bjet"], axis=1)
    events = events[events["n_bjets"] <= 1]
    tracker.update("B-jet Veto", events)

    # --- Cut 6: Lepton/Tau Veto ---
    # Tau Veto
    n_tau = ak.sum(events["AnalysisTauJetsAuxDyn_JetDeepSetTight"] == 1, axis=1)
    events = events[n_tau == 0]
    tracker.update("Tau Veto", events)
    
    # Count Leptons for Region Splitting
    n_ele = ak.sum(events["AnalysisElectronsAuxDyn_DFCommonElectronsLHTight"] == 1, axis=1)
    
    mu_qual = (events["AnalysisMuonsAuxDyn_quality"] == 8) | (events["AnalysisMuonsAuxDyn_quality"] == 9) # Medium
    mu_type = (events["AnalysisMuonsAuxDyn_muonType"] == 0) # Combined
    n_mu = ak.sum(mu_qual & mu_type, axis=1)
    
    events["n_ele"] = n_ele
    events["n_mu"] = n_mu
    
    tracker.update("Preselection Complete", events)
    return events

# === 3. REGION SPLITTING & SAVING ===
def split_and_save(events, process_name, tracker, batch_suffix):
    """
    Splits events into regions and saves them as Parquet files with a batch suffix.
    """
    if len(events) == 0: return

    # Define Masks
    jet_pt = events["AnalysisJetsAuxDyn_pt"] / 1000.0
    ht = ak.sum(jet_pt[abs(events["AnalysisJetsAuxDyn_eta"]) < 2.8], axis=1)
    
    kin_mask_sr = (events["met_recalc_pt"] > 600) & (ht > 600)
    kin_mask_cr = (events["met_recalc_pt"] > 250) & (events["met_recalc_pt"] <= 600) & (ht <= 600)
    kin_mask_qcd = (events["met_recalc_pt"] <= 600) & (ht <= 600)

    regions = {
        "SR": events[kin_mask_sr & (events["n_ele"] == 0) & (events["n_mu"] == 0) & (events["n_bjets"] <= 1)],
        # --- Original Muon Regions ---
        "CR1mu": events[kin_mask_cr & (events["n_mu"] == 1) & (events["n_ele"] == 0) & (events["n_bjets"] == 0)],
        "CR1mub": events[kin_mask_cr & (events["n_mu"] == 1) & (events["n_ele"] == 0) & (events["n_bjets"] == 1)],
        "CR2mu": events[kin_mask_cr & (events["n_mu"] == 2) & (events["n_ele"] == 0) & (events["n_bjets"] == 0)],

        # --- New Electron Regions (Doubles W/Top Stats) ---
        "CR1ele": events[kin_mask_cr & (events["n_ele"] == 1) & (events["n_mu"] == 0) & (events["n_bjets"] == 0)],
        "CR1eleb": events[kin_mask_cr & (events["n_ele"] == 1) & (events["n_mu"] == 0) & (events["n_bjets"] == 1)],
        "CR2ele": events[kin_mask_cr & (events["n_ele"] == 2) & (events["n_mu"] == 0) & (events["n_bjets"] == 0)],

        # --- Different Flavor (Pure ttbar) ---
        "CR_emu": events[kin_mask_cr & (events["n_mu"] == 1) & (events["n_ele"] == 1)],

        # --- The Multijet/QCD Template
        "CR0L_Inclusive": events[kin_mask_qcd & (events["n_ele"] == 0) & (events["n_mu"] == 0)]
    }

    # Update Tracker with Final Regions
    for reg, data in regions.items():
        tracker.update(f"Region: {reg}", data)
        
        # SAVE TO DISK (Using batch_suffix to prevent overwriting)
        if len(data) > 0:
            save_path = os.path.join(OUTPUT_DIR, reg)
            os.makedirs(save_path, exist_ok=True)
            
            # Creates: Regions/SR/ttbar_batch0.parquet, ttbar_batch1.parquet...
            file_name = os.path.join(save_path, f"{process_name}{batch_suffix}.parquet")
            
            ak.to_parquet(data, file_name)
            # print(f"Saved {len(data)} events to {file_name}") # Optional: Comment out to reduce noise

# === 4. MAIN PIPELINE ===
def process_full_dataset(process_name):
    print(f"--> Processing {process_name}...")
    tracker = CutflowTracker()
    batch_counter = 0 # Unique ID for output files
    
    # Load all subprocesses
    subprocesses = list(METADATA[process_name].keys())
    
    for subp in subprocesses:
        ttbar_path = os.path.join(BASE_PATH, process_name)
        subp_dir = os.path.join(ttbar_path, f"mc20_13TeV_MC_{subp}")
        
        if not os.path.exists(subp_dir): continue
        
        files = sorted([os.path.join(subp_dir, f) for f in os.listdir(subp_dir) if f.endswith(".root")])
        if not files: continue
        
        try:
            # === MEMORY FIX: Use iterate instead of concatenate ===
            # This loads data in chunks (BATCH_SIZE) rather than all at once.
            for events in uproot.iterate(
                [f + ":CollectionTree" for f in files], 
                BRANCHES, 
                library="ak", 
                step_size=BATCH_SIZE
            ):
                
                # Weight Calculation
                meta = METADATA[process_name][subp]
                norm = (meta['xsec_pb'] * LUMI) / (meta['sum_w'] * REDUCTION)
                events["weight_phys"] = events["EventInfoAuxDyn_mcEventWeights"][:, 0] * norm
                # events["mc_subprocess"] = ak.full_like(events["weight_phys"], subp, dtype=str)
                events["mc_subprocess"] = ak.Array([subp] * len(events))
                # CLEAN (Updates tracker internally, accumulations work fine)
                cleaned_events = apply_preselection(events, tracker)
                
                # SPLIT & SAVE (Pass batch_counter to create unique files)
                split_and_save(cleaned_events, process_name, tracker, f"_batch{batch_counter}")
                
                batch_counter += 1
                
        except Exception as e:
            print(f"Error in {subp}: {e}")

    # Save Cutflow CSV (Tracker sums up all batches automatically)
    tracker.save_csv(process_name, OUTPUT_DIR)
    print(f"<-- Finished {process_name}")

if __name__ == "__main__":
    processes = ["ttbar", "Diboson", "Single_top", "Multijet", "Wjets", "Zjets"]
    
    start_time = time.time()
    now = datetime.datetime.now()
    cpu_cores = multiprocessing.cpu_count()
    
    print("="*60)
    print(f"  HEP DATA PROCESSING PIPELINE (BATCHED)")
    print(f"  Date:       {now.strftime('%Y-%m-%d')}")
    print(f"  System:     {cpu_cores} CPU Cores Detected")
    print(f"  Config:     {MAX_WORKERS} Workers | Batch Size: {BATCH_SIZE}")
    print("="*60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_full_dataset, processes)

    elapsed = time.time() - start_time
    print("="*60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Total Duration: {elapsed/60:.2f} minutes")
    print("="*60)