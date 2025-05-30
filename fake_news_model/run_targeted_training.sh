#!/bin/bash


GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting targeted fake news detection training${NC}"

mkdir -p fake_news_model/targeted_results


echo -e "${YELLOW}Checking and installing required packages...${NC}"
pip install -q transformers datasets torch scikit-learn matplotlib seaborn tqdm sentencepiece


echo -e "${GREEN}Running targeted training with improved label handling${NC}"
echo -e "${YELLOW}This will focus on fixing the label format issues that caused zero metrics${NC}"


python fake_news_model/targeted_tuning.py


if [ $? -eq 0 ]; then
    echo -e "${GREEN}Targeted training completed successfully!${NC}"
else
    echo -e "${RED}Training failed. Check logs for details.${NC}"
    exit 1
fi


echo -e "${GREEN}Results available at:${NC}"
echo -e "- Training logs: ${YELLOW}fake_news_model/targeted_results/training_log.txt${NC}"
echo -e "- Training summary: ${YELLOW}fake_news_model/targeted_results/training_summary.txt${NC}"
echo -e "- Metrics: ${YELLOW}fake_news_model/targeted_results/final_metrics.json${NC}"
echo -e "- Visualizations: ${YELLOW}fake_news_model/targeted_results/eval_metrics.png${NC}"


if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Opening training visualizations...${NC}"
    open fake_news_model/targeted_results/training_loss.png
    open fake_news_model/targeted_results/eval_metrics.png 2>/dev/null || true
fi

echo -e "${GREEN}Training process completed!${NC}"

exit 0 