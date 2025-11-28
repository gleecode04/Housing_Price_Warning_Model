
import sys
import os
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(SCRIPT_DIR / 'scripts' / 'eda'))

def main():
    
    
    import importlib.util
    eda_path = SCRIPT_DIR / 'scripts' / 'eda' / 'run_eda.py'
    spec = importlib.util.spec_from_file_location("run_eda", eda_path)
    eda_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda_module)
    df = eda_module.main()
    
    if df is None or df.empty:
        print("\nERROR: EDA failed!")
        return
    
    print("eda done")
    
   
    prep_path = SCRIPT_DIR / 'scripts' / 'preprocessing' / 'run_preprocessing.py'
    spec = importlib.util.spec_from_file_location("run_preprocessing", prep_path)
    prep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep_module)
    result = prep_module.main()
    
    if result is None or result[0] is None:
        print("\nERROR: Preprocessing failed!")
        return

    df_preprocessed, info = result
    if df_preprocessed.empty:
        print("\nERROR: Preprocessing produced empty dataset!")
        return
    
    print("preprocessing done")
    
    
if __name__ == "__main__":
    main()

