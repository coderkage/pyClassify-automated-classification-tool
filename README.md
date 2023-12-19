# pyClassify - Automated ML-based Tool for Classification Problems

This project aims to develop a comprehensive Python script that automates the process of running (binary or multi-class) classification problems on any given input data matrix in the form of a standard Feature x Instance matrix (.csv) file.

## Project Structure

The project consists of multiple Python scripts, each responsible for a specific step in the machine learning pipeline. The main script orchestrates the execution of these scripts, ensuring a seamless and automated workflow.

### Project Components

1. **Preprocessing:**
   - This script handles the missing value and string conversion of the given main input file.
   - User input: classification data in .csv format.
   - Output: Preprocessed data matrix.

2. **Normalization/Standardization:**
   - This script handles the normalization or standardization of the input data.
   - User input: Type of normalization or standardization.
   - Output: Normalized or standardized data matrix.

3. **Feature Selection:**
   - This script performs feature selection on the preprocessed data.
   - User input: Feature selection method and parameters.
   - Output: Data matrix with selected features.

4. **Cross-Validation Script:**
   - Implements cross-validation on the data.
   - User input: Number of folds for cross-validation.
   - Output: Cross-validated performance metrics.

5. **Machine Learning Modeling Script:**
   - Executes the machine learning modeling for classification.
   - User input: Classification algorithm and hyperparameters.
   - Output: Trained machine learning model.

6. **Prediction Script:**
   - Evaluates the predictive capability of the model on a blind dataset.
   - Output: Accuracy and other performance metrics on the blind dataset.

7. **Main Script:**
   - Orchestrates the execution of the above scripts.
   - User input: File path of the input data matrix (.csv).
   - Output: Generates plots, heatmaps, and prints performance metrics in a PDF file.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/automated-ml-classification.git
    cd automated-ml-classification
    ```

2. Install dependencies:

    ```bash
    pip install fpdf
    ```

3. Run the main script:

    ```bash
    python main.py
    ```

4. Follow the prompts to provide input options for each step of the pipeline.

5. Check the output PDF file for performance metrics and plots.

## Notes

- Ensure that the input data matrix is in the required format (Feature x Instance matrix in .csv format).

Feel free to contribute, report issues, or suggest improvements. Happy classifying!

