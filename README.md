# CFIUS Review Classification Prototype

This repository contains the prototype for a data pipeline designed to identify and classify transactions subject to CFIUS review from 8-K filings. The prototype includes data collection, text extraction, and classification using a transformer model.

## Directory Structure

- **data/**: Directory for storing data files.
- **prototype.ipynb**: Jupyter notebook containing the full implementation of the prototype.
- **prototype.pdf**: PDF version of the Jupyter notebook.
- **README.md**: This README file.
- **requirements.txt**: List of Python dependencies required to run the prototype.
- **utils.py**: Utility functions used in the prototype.

## Overview

The script implements the following data pipeline:

1. **Scrape EDGAR for 8-K Filings**:
   - Identify 8-K filings within a specified timeframe that contain the keyword ‘CFIUS’.
   - Extract relevant company information and format the URLs for scraping the 8-K forms.

2. **Extract Relevant Texts**:
   - Retrieve and extract texts surrounding the keyword ‘CFIUS’ from the identified 8-K filings.

3. **Classify Extracted Texts**:
   - Use a transformer model (RoBERTa) to classify the extracted texts into one of three categories:
     - CFIUS review announced
     - Notices of CFIUS approvals
     - Notices of CFIUS denials

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Prototype

To run the prototype, execute the `prototype.ipynb` notebook in Jupyter. The notebook provides a step-by-step implementation of the data pipeline, from data collection to text extraction and classification.

## Contact

For any questions or further information, please contact Jiayan Li at jiayanjoanneli@gmail.com.