import os
import ROOT
import json
import sys
import datetime
import argparse
import math
import random

# ------------------ ROOT CONFIGURATION ------------------
# Suppress ROOT info/warning messages
ROOT.gErrorIgnoreLevel = ROOT.kError

# ------------------ ROOT CONFIGURATION ------------------
# Suppress ROOT info/warning messages
ROOT.gErrorIgnoreLevel = ROOT.kError

# Removed ATLAS-specific xAOD types to prevent JIT compilation crashes
types_to_generate = [
    "vector<float>",
    "vector<vector<float>>",
    "vector<int>",
    "vector<short>",
    "vector<double>",
    "vector<unsigned int>",
    "vector<unsigned char>",
    "vector<char>",
    "vector<string>",
    "vector<ULong64_t>"
]

for t in types_to_generate:
    ROOT.gInterpreter.GenerateDictionary(f"ROOT::VecOps::RVec<{t}>", "vector;vector;ROOT/RVec.hxx")

# ------------------ ARGUMENT PARSING ------------------
parser = argparse.ArgumentParser(description="Process ATLAS MC datasets")
parser.add_argument("-process", type=str, required=True,
                    help="Process to run (e.g. Wjets, Zjets, ttbar, Single_top, Multijet, Diboson)")
args = parser.parse_args()
process_to_run = args.process

# ------------------ LOGGING ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

today = datetime.date.today().strftime("%Y-%m-%d")
existing_logs = [f for f in os.listdir(log_dir) if f.startswith(today)]
run_number = len(existing_logs) + 1
log_file = os.path.join(log_dir, f"{today}_run{run_number}_{process_to_run}.txt")

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
print(f"Logging to {log_file}\n")

# ------------------ ATLAS MC INFO ------------------
atlas_info = {
    "Wjets": {
        "jsons": [
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
    "Zjets": {
        "jsons": [
            "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_BFilter_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2211_Znunu_pTV2_CVetoBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2214_Ztautau_maxHTpTV2_CFilterBVeto_file_index.json",
            "mc20_13TeV_MC_Sh_2214_Ztautau_maxHTpTV2_CVetoBVeto_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
    "ttbar": {
        "jsons": [
            "mc20_13TeV_MC_PhPy8EG_A14_ttbar_hdamp258p75_nonallhad_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_ttbar_hdamp258p75_allhad_file_index.json"
        ],
        "file": "ATLAS_ttbar.json"
    },
    "Single_top": {
        "jsons": [
            "mc20_13TeV_MC_PowhegPythia8EvtGen_A14_singletop_schan_lept_top_file_index.json",
            "mc20_13TeV_MC_PowhegPythia8EvtGen_A14_singletop_schan_lept_antitop_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_tchan_BW50_lept_top_file_index.json",
            "mc20_13TeV_MC_PhPy8EG_A14_tchan_BW50_lept_antitop_file_index.json"
        ],
        "file": "ATLAS_ttbar.json"
    },
    "Multijet": {
        "jsons": [
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ1WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ2WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ4WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ5WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ6WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ8WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ9WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ10WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ11WithSW_file_index.json",
            "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ12WithSW_file_index.json"
        ],
        "file": "ATLAS_QCD.json"
    },
    "Diboson": {
        "jsons": [
            "mc20_13TeV_MC_Sh_2211_WlvZqq_file_index.json",
            "mc20_13TeV_MC_Sh_2211_WqqZvv_file_index.json",
            "mc20_13TeV_MC_Sh_2211_ZqqZvv_file_index.json",
            "mc20_13TeV_MC_Sh_2211_WlvWqq_file_index.json"
        ],
        "file": "ATLAS_boson.json"
    },
}

# ------------------ LOAD MC METADATA ------------------
def load_metadata(file):
    with open(file, "r") as f:
        return json.load(f)

def get_root_links_from_json(meta, json_name):
    links = []
    for meta_run in meta["metadata"]["_file_indices"]:
        if meta_run["key"] == json_name:
            for root_file in meta_run["files"]:
                links.append(root_file["uri"])
    return links

# ------------------ PREPARE DATASETS ------------------
all_processes = {}
for process, info in atlas_info.items():
    meta = load_metadata(info["file"])
    all_processes[process] = {
        json_name.replace("_file_index.json", ""): get_root_links_from_json(meta, json_name)
        for json_name in info["jsons"]
    }

# ------------------ REDUCE ROOT ------------------
path_reduce_root = "NRAD_Dataset/MC/reduce_root"
os.makedirs(path_reduce_root, exist_ok=True)

# ------------------ BRANCH SELECTION ------------------
rel_branches = [

    # --- Global Event & MET ---
    "MET_Core_AnalysisMETAuxDyn.mpx",  
    "MET_Core_AnalysisMETAuxDyn.mpy",  
    "MET_Core_AnalysisMETAuxDyn.sumet",


    # --- Event Info ---
    "EventInfoAuxDyn.mcEventWeights",   
    "EventInfoAuxDyn.runNumber",        
    "EventInfoAuxDyn.eventNumber",      
    "EventInfoAuxDyn.lumiBlock",       
    "EventInfoAuxDyn.lumiFlags", 
    "EventInfoAuxDyn.muonFlags",
    "EventInfoAuxDyn.PileupWeight_NOSYS",
    "EventInfoAuxDyn.coreFlags",
    "EventInfoAuxDyn.beamStatus",

    # --- Analysis Electrons --- 
    "AnalysisElectronsAuxDyn.ambiguityType",
    "AnalysisElectronsAuxDyn.author",
    "AnalysisElectronsAuxDyn.pt",    
    "AnalysisElectronsAuxDyn.eta",   
    "AnalysisElectronsAuxDyn.phi",   
    "AnalysisElectronsAuxDyn.m",     
    "AnalysisElectronsAuxDyn.charge",
    "AnalysisElectronsAuxDyn.DFCommonElectronsECIDS",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHVeryLoose",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHLoose",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHLooseBL",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHMedium",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHTight",
    "AnalysisElectronsAuxDyn.firstEgMotherPdgId",
    "AnalysisElectronsAuxDyn.OQ",


    # --- Small-R Jets ---
    "AnalysisJetsAuxDyn.ActiveArea4vec_eta",
    "AnalysisJetsAuxDyn.ActiveArea4vec_m",
    "AnalysisJetsAuxDyn.ActiveArea4vec_phi",
    "AnalysisJetsAuxDyn.ActiveArea4vec_pt",

    "AnalysisJetsAuxDyn.pt",                    
    "AnalysisJetsAuxDyn.eta",                   
    "AnalysisJetsAuxDyn.phi",                   
    "AnalysisJetsAuxDyn.m",                     
    "AnalysisJetsAuxDyn.NNJvtPass",             
    "AnalysisJetsAuxDyn.SumPtChargedPFOPt500",  
    "AnalysisJetsAuxDyn.EnergyPerSampling",     
    "AnalysisJetsAuxDyn.ConeTruthLabelID",
    "AnalysisJetsAuxDyn.EMFrac",
    "AnalysisJetsAuxDyn.GhostMuonSegmentCount",
    "AnalysisJetsAuxDyn.JVFCorr",
    "AnalysisJetsAuxDyn.NumTrkPt500",
    "AnalysisJetsAuxDyn.DFCommonJets_QGTagger_NTracks",

    
    # --- Large-R Jets ---
    "AnalysisLargeRJetsAuxDyn.pt",        
    "AnalysisLargeRJetsAuxDyn.eta",       
    "AnalysisLargeRJetsAuxDyn.phi",       
    "AnalysisLargeRJetsAuxDyn.m",         
    "AnalysisLargeRJetsAuxDyn.Tau1_wta",  
    "AnalysisLargeRJetsAuxDyn.Tau2_wta",  
    "AnalysisLargeRJetsAuxDyn.Tau3_wta",  
    "AnalysisLargeRJetsAuxDyn.D2",        
    "AnalysisLargeRJetsAuxDyn.C2",        
    
    # --- Muons & InDet Tracks ---
    "AnalysisMuonsAuxDyn.pt",                     
    "AnalysisMuonsAuxDyn.eta",                    
    "AnalysisMuonsAuxDyn.phi",                    
    "AnalysisMuonsAuxDyn.quality",          
    "AnalysisMuonsAuxDyn.muonType",
    "AnalysisMuonsAuxDyn.charge",
    "AnalysisMuonsAuxDyn.EnergyLoss",
    "AnalysisMuonsAuxDyn.energyLossType",

    # --- AnalysisPhotons ---
    "AnalysisPhotonsAuxDyn.pt",
    "AnalysisPhotonsAuxDyn.eta",
    "AnalysisPhotonsAuxDyn.phi",
    "AnalysisPhotonsAuxDyn.m",
    "AnalysisPhotonsAuxDyn.OQ",

    "AnalysisPhotonsAuxDyn.author",
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsCleaning",
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMTight",
    "AnalysisPhotonsAuxDyn.DFCommonPhotonsIsEMLoose",

    # --- AnalysisTauJets ---
    "AnalysisTauJetsAuxDyn.charge",
    "AnalysisTauJetsAuxDyn.ptFinalCalib",
    "AnalysisTauJetsAuxDyn.etaFinalCalib",
    "AnalysisTauJetsAuxDyn.EleRNNLoose_v1",
    "AnalysisTauJetsAuxDyn.EleRNNMedium_v1",
    "AnalysisTauJetsAuxDyn.EleRNNTight_v1",
    "AnalysisTauJetsAuxDyn.isTauFlags",
    "AnalysisTauJetsAuxDyn.JetDeepSetVeryLoose",
    "AnalysisTauJetsAuxDyn.JetDeepSetLoose",
    "AnalysisTauJetsAuxDyn.JetDeepSetMedium",
    "AnalysisTauJetsAuxDyn.JetDeepSetTight",
    
    "AnalysisTauJetsAuxDyn.eta",
    "AnalysisTauJetsAuxDyn.pt",
    "AnalysisTauJetsAuxDyn.phi",              
    "AnalysisTauJetsAuxDyn.m",                
    "AnalysisTauJetsAuxDyn.PanTau_DecayMode",     
    "AnalysisTauJetsAuxDyn.NNDecayMode",

    # --- Flavor Tagging (DL1d) ---
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu", 
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc", 
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"  
]

def reduce_root(process, dataset, link, c):
    chain = ROOT.TChain("CollectionTree")
    chain.Add(link)
    df = ROOT.RDataFrame(chain)
    outdir = f"{path_reduce_root}/{process}/{dataset}"
    os.makedirs(outdir, exist_ok=True)
    df.Snapshot("CollectionTree", f"{outdir}/root_{c}.root", rel_branches)

# ------------------ MAIN LOOP ------------------
if process_to_run not in all_processes:
    print(f"ERROR: Process '{process_to_run}' not found! Available: {list(all_processes.keys())}")
    sys.exit(1)

datasets = all_processes[process_to_run]
print(f"\n=== Processing Process {process_to_run} ===")

for dataset_name, links in datasets.items():
    print(f"\n--- Dataset {dataset_name}: {len(links)} files ---")

    n_select = math.ceil(len(links))
    selected_links = random.sample(links, n_select)
    print(f"Selected {n_select} files")

    reduce_outdir = f"{path_reduce_root}/{process_to_run}/{dataset_name}"
    
    for c, link in enumerate(selected_links):
        print(f"[{c+1}/{n_select}] Processing {link}")
        reduce_root(process_to_run, dataset_name, link, c)

print(f"\n✅ Finished processing {process_to_run}. Results saved in {path_reduce_root}/{process_to_run}/")