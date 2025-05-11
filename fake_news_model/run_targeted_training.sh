#!/bin/bash

# Run targeted training for fake news detection model
# This script runs a focused training approach to get accurate metrics

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting targeted fake news detection training${NC}"

# Create required directories
mkdir -p fake_news_model/targeted_results

# Install required packages if needed
echo -e "${YELLOW}Checking and installing required packages...${NC}"
pip install -q transformers datasets torch scikit-learn matplotlib seaborn tqdm sentencepiece

# Run the targeted training approach
echo -e "${GREEN}Running targeted training with improved label handling${NC}"
echo -e "${YELLOW}This will focus on fixing the label format issues that caused zero metrics${NC}"

# Run the training script
python fake_news_model/targeted_tuning.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Targeted training completed successfully!${NC}"
else
    echo -e "${RED}Training failed. Check logs for details.${NC}"
    exit 1
fi

# Display results location
echo -e "${GREEN}Results available at:${NC}"
echo -e "- Training logs: ${YELLOW}fake_news_model/targeted_results/training_log.txt${NC}"
echo -e "- Training summary: ${YELLOW}fake_news_model/targeted_results/training_summary.txt${NC}"
echo -e "- Metrics: ${YELLOW}fake_news_model/targeted_results/final_metrics.json${NC}"
echo -e "- Visualizations: ${YELLOW}fake_news_model/targeted_results/eval_metrics.png${NC}"

# Open visualizations if on a Mac
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Opening training visualizations...${NC}"
    open fake_news_model/targeted_results/training_loss.png
    open fake_news_model/targeted_results/eval_metrics.png 2>/dev/null || true
fi

echo -e "${GREEN}Training process completed!${NC}"

exit 0 