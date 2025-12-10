# 🧠 Epilepsy Prediction: EEG-Based Seizure Forecasting with Temporal Attention Networks

## 📊 Project Overview
This project focuses on epileptic seizure prediction tasks, utilizing the publicly available EEG dataset from Boston Children's Hospital. Through innovative time series analysis methods, it aims to achieve accurate pre-ictal period identification. By selecting optimal EEG channels based on power spectral density analysis, implementing data balancing strategies to address class imbalance, and leveraging advanced time series models like TimesNet, the project establishes a high-precision epilepsy prediction system.

## 🎯 Research Objectives
- **Optimal Channel Selection**: Identify the best channel combinations from 6 candidate channels based on power spectral density analysis
- **Data Balancing Optimization**: Address the severe imbalance between interictal and preictal period data
- **Temporal Pattern Mining**: Utilize TimesNet to capture multi-scale periodic patterns
- **Early Warning System**: Achieve reliable prediction 30 minutes before seizure onset
- **Reproducible Benchmark**: Provide a standardized pipeline for epilepsy prediction research

## 🔬 Experiment Design
### Challenges in Epilepsy Prediction
- **Data Imbalance**: Interictal data significantly outnumbers preictal data
- **Signal Complexity**: EEG signals contain substantial noise with significant individual variations
- **Temporal Sensitivity**: Requires sufficient lead time for effective early warnings
- **Channel Selection**: Different brain regions show varying sensitivity to epileptic activity

### Dataset that I SELECT 
- **Data Source**: Publicly available EEG dataset from Boston Children's Hospital
- **Number of Channels**: 6 bipolar channel candidates
- **Time Period Definitions**:
  - **Preictal Period**: 30 minutes before seizure onset
  - **Ictal Period**: During clinical seizure episodes
  - **Interictal Period**: All remaining time periods
