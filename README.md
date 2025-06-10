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

| Argument            | Type    | Description                                                                                  | Default Value                                                    |
|---------------------|---------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `--zip_path`        | str     | Path to input zip file containing repertoires.                                               | `public/towlerton.zip`                                             |
| `--download`        | flag    | Download the default zip file if not present.                                                | (not set)                                                        |
| `--zip_url`         | str     | URL to download zip file (used with `--download`).                                           | *see code*                                                       |
| `--model_folder`    | str     | Path to model folder.                                                                        | `public/XSTUDY_ALL_FEATURE_L1_v4e`                                 |
| `--model_name`      | str     | Model name.                                                                                  | `XSTUDY_ALL_FEATURE_L1_v4e`                                      |
| `--model_calibration`| str    | Model calibration name.                                                                      | `XSTUDY_ALL_FEATURE_L1_v4e_HS2`                                  |
| `--truth_file`      | str     | Path to truth file for scoring.                                                              | `public/XSTUDY_ALL_FEATURE_L1_v4e/sample_hla_x_towlerton.csv`      |
| `--output_folder`   | str     | Folder to write outputs.                                                                     | `outputs`                                                        |
| `--n`               | int     | Maximum number of files to process from zip.                                                 | `5`                                                              |
| `--run`             | str     | Run name for output files.                                                                   | `towlerton_demo10`                                               |
| `--v_col`           | str     | V column name in input files.                                                                | `vMaxResolved`                                                   |
| `--cdr3_col`        | str     | CDR3 column name in input files.                                                             | `aminoAcid`                                                      |
| `--templates_col`   | str     | Templates column name in input files.                                                        | `count (templates/reads)`                                        |

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