# TCR2HLAi

A command-line tool for running HLA prediction on TCR repertoire zip files using pre-trained models.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd TCR2HLAi
   ```

2. **(Optional) Create and activate a Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the command-line program:**
   ```bash
   python TCR2HLAi.py --download
   ```
   This will automatically download the default example data if not present and run the pipeline.

## Input Arguments

You can customize the run using the following arguments:

| Argument            | Type    | Default Value                                                    | Description                                                                                  |
|---------------------|---------|------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `--zip_path`        | str     | `data/towlerton.zip`                                             | Path to input zip file containing repertoires.                                               |
| `--download`        | flag    | (not set)                                                        | Download the default zip file if not present.                                                |
| `--zip_url`         | str     | *see code*                                                       | URL to download zip file (used with `--download`).                                           |
| `--model_folder`    | str     | `data/XSTUDY_ALL_FEATURE_L1_v4e`                                 | Path to model folder.                                                                        |
| `--model_name`      | str     | `XSTUDY_ALL_FEATURE_L1_v4e`                                      | Model name.                                                                                  |
| `--model_calibration`| str    | `XSTUDY_ALL_FEATURE_L1_v4e_HS2`                                  | Model calibration name.                                                                      |
| `--truth_file`      | str     | `data/XSTUDY_ALL_FEATURE_L1_v4e/sample_hla_x_towlerton.csv`      | Path to truth file for scoring.                                                              |
| `--output_folder`   | str     | `outputs`                                                        | Folder to write outputs.                                                                     |
| `--n`               | int     | `5`                                                              | Maximum number of files to process from zip.                                                 |
| `--run`             | str     | `towlerton_demo10`                                               | Run name for output files.                                                                   |
| `--v_col`           | str     | `vMaxResolved`                                                   | V column name in input files.                                                                |
| `--cdr3_col`        | str     | `aminoAcid`                                                      | CDR3 column name in input files.                                                             |
| `--templates_col`   | str     | `count (templates/reads)`                                        | Templates column name in input files.                                                        |

**Example:**
```bash
python TCR2HLAi.py --zip_path mydata.zip --n 10 --run myrun
```

## Notes

- If you do not have the example data, use `--download` to fetch it automatically.
- Outputs will be written to the folder specified by `--output_folder`.

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies.

---