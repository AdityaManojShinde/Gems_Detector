 # Tablet Automatic Optical Inspection System

An automated vision-based inspection system for detecting and classifying tablet defects. The system uses image processing and computer vision to inspect tablets for issues.

## Features

- Automated tablet image inspection
- Defect detection and classification
- Identification of damaged, malformed, or discolored tablets
- Reduced manual inspection effort
- Consistent and repeatable quality control
- Support for real-time or batch inspection workflows

## Project Objective

The objective of this project is to provide a reliable, non-contact quality inspection solution for pharmaceutical tablets. Captured tablet images are processed to extract visual features and determine whether each tablet meets the required quality standards.

## System Workflow

1. Capture tablet images using a camera or imaging device.
2. Preprocess images to reduce noise and improve quality.
3. Segment the tablet from its background.
4. Analyze visual characteristics such as shape, size, color, and surface defects.
5. Classify the tablet as acceptable or defective.
6. Display or store inspection results for quality monitoring.

## Applications

- Pharmaceutical manufacturing
- Tablet production lines
- Laboratory quality assurance
- Automated packaging inspection
- Production defect monitoring

## Requirements

- Python 3.x
- Camera or image dataset for tablet inspection
- Computer vision and image-processing libraries required by the implementation
- A computer capable of processing captured images

## Installation

Clone the repository and install the project dependencies:

```bash
git clone <repository-url>
cd Gems_Detector
pip install -r requirements.txt
```

## Usage

Run the project using the main application file provided in the repository:

```bash
python <main-file>.py
```

Provide the required image, camera, or dataset input and review the generated inspection results.

## Expected Output

The system reports the inspection status of each tablet and may provide annotated images, defect labels, confidence scores, or summary statistics depending on the implementation.

## Future Enhancements

- Integration with industrial cameras and conveyor systems
- Real-time production-line inspection
- Improved defect classification using machine learning
- Automatic rejection of defective tablets
- Inspection reports and database integration

