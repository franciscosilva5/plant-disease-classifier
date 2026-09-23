# Plant Disease Classifier

Computer vision project for classifying bean leaf images into three categories:

- Angular Leaf Spot
- Bean Rust
- Healthy

The project uses transfer learning with MobileNetV3 Small and includes model evaluation, confidence-aware predictions, Grad-CAM visual explanations, automated testing and Streamlit deployment.

## Live Demo

https://plant-disease-classifier-vercetius.streamlit.app/

## Overview

The application accepts a leaf image and returns:

- predicted class
- confidence score
- top-3 class probabilities
- confidence warning when the prediction is below the application threshold
- Grad-CAM visualization showing which image regions influenced the prediction

The application always shows the model prediction, but predictions below 70% confidence are explicitly marked as low confidence.

## Model

The final model uses:

- MobileNetV3 Small
- ImageNet pretrained weights
- transfer learning
- PyTorch
- three output classes

Transfer learning was selected because the dataset is relatively small compared with large-scale computer-vision datasets.

## Evaluation

Final test-set results:

| Metric | Result |
| --- | ---: |
| Test Accuracy | 90.62% |
| Macro F1 | 90.77% |

Per-class F1:

| Class | F1 |
| --- | ---: |
| Angular Leaf Spot | 86.36% |
| Bean Rust | 88.37% |
| Healthy | 97.56% |

The confusion matrix is available below:

![Confusion Matrix](images/confusion-matrix.png)

These results are specific to the project dataset and should not be interpreted as production-level disease-diagnosis performance.

## Explainability

Grad-CAM is used to visualize image regions that influenced the model prediction.

Example:

![Grad-CAM](images/gradcam-test.png)

Grad-CAM provides qualitative interpretability but does not prove that the model is using biologically correct visual features.

## Design Decisions and Trade-offs

**Transfer learning was preferred over training from scratch.** MobileNetV3 provides strong visual features while requiring substantially less data and training time than a custom CNN trained from random initialization.

**MobileNetV3 Small was chosen for deployment efficiency.** The final model is compact enough for an interactive Streamlit application while still providing strong classification performance.

**Confidence is shown separately from the predicted class.** The application does not hide low-confidence predictions. Instead, predictions below 70% are shown with a warning so uncertainty remains visible to the user.

**The 70% confidence threshold is an application rule, not a calibrated guarantee of correctness.** Neural-network softmax probabilities can be overconfident.

**Grad-CAM is used for transparency rather than as proof of correctness.** It helps inspect model attention but should not be interpreted as a clinical or agricultural diagnosis explanation.

## Automated Tests

The project includes an automated inference test that verifies:

- the trained model loads correctly
- inference completes successfully
- the predicted class belongs to the known class set
- confidence remains between 0 and 1
- the top-3 probability output is valid
- probability values sum correctly

Run:

pytest -q

GitHub Actions automatically runs the test on pushes and pull requests to main.

## Application Screenshots

### Angular Leaf Spot

![Angular Leaf Spot Prediction](images/app/angular-leaf-spot-result.png)

### Bean Rust

![Bean Rust Prediction](images/app/bean-rust-result.png)

### Healthy

![Healthy Prediction](images/app/healthy-result.png)

## Architecture

Leaf Image
→ Image preprocessing
→ MobileNetV3 Small
→ Class probabilities
→ Prediction + confidence
→ Confidence policy
→ Grad-CAM explanation
→ Streamlit interface

The inference logic is separated from the Streamlit interface so the model behavior can be tested independently.

## Run Locally

Clone the repository:

git clone https://github.com/franciscosilva5/plant-disease-classifier.git

Enter the project:

cd plant-disease-classifier

Create a virtual environment:

python3.12 -m venv .venv

Activate it:

source .venv/bin/activate

Install dependencies:

python -m pip install -r requirements.txt

Run the application:

streamlit run app/app.py

Run the tests:

pytest -q

## Tech Stack

- Python
- PyTorch
- torchvision
- MobileNetV3
- Pillow
- NumPy
- Streamlit
- Grad-CAM
- pytest
- GitHub Actions

## Limitations

This is a portfolio computer-vision project rather than a production agricultural diagnosis system.

Important limitations include:

- limited number of disease classes
- performance depends on the distribution and quality of the training images
- real-world lighting, backgrounds, camera quality and leaf orientation may differ from the dataset
- confidence values are not statistically calibrated probabilities
- Grad-CAM is qualitative
- the model has not been validated for professional agricultural decision-making
- no continual learning or production monitoring is implemented

A production system would require broader field data, external validation, confidence calibration, monitoring, domain-expert review and evaluation across different devices and environmental conditions.

## Future Improvements

- external validation on unseen field images
- confidence calibration
- additional plant diseases and species
- stronger augmentation experiments
- model monitoring
- mobile deployment

## License

See the repository license for project licensing information.
