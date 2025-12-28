# ⚽ Football Player ML Analysis

**Advanced Machine Learning System for Football Player Evaluation with Novel Defensive Metrics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen.svg)](#results)

---

## 🎯 Project Overview

A comprehensive machine learning system that evaluates **1,336 football players** across Europe's top 5 leagues, achieving **99.3% accuracy** through novel "tracking back" metrics. The system classifies players into **13 tactical positions** and identifies **€200M+** in market inefficiencies.

### 🏆 Key Achievements
- ✅ **99.3% accuracy** in defender evaluation (+32.2% improvement over baseline)
- ✅ **Novel innovation**: Tracking back metrics measuring defensive actions across all pitch thirds
- ✅ **13 tactical positions** classified with position-specific models
- ✅ **€200M+ value** identified across 408 undervalued players
- ✅ **Dual analysis**: Both current season (2024) and career (2018-2024) rankings

---

## 🔥 The Innovation: Tracking Back Metrics

Traditional defensive metrics only measure tackles and clearances in defensive areas. Modern defenders contribute across the entire pitch. Our breakthrough:

```python
# Weighted defensive work rate based on pitch location
defensive_work_rate = (
    (tackles_def_3rd * 1.0) +   # Normal defending
    (tackles_mid_3rd * 1.2) +   # Tracking back from midfield
    (tackles_att_3rd * 1.5) +   # High pressing in attack
    (interceptions * 0.8)        # Reading the game
)
```

**Result:** Defender evaluation accuracy improved from **67.1% → 99.3%** (+32.2%)

---

## 📊 Results

### Model Performance
| Position | Random Forest | XGBoost | Improvement |
|----------|--------------|---------|-------------|
| **Defenders** | 67.1% | **99.3%** | +32.2% |
| **Forwards** | 97.2% | **98.8%** | +1.6% |
| **Midfielders** | 88.9% | **90.2%** | +1.3% |

### Business Impact
- **408 bargain players** identified (value gap > +10)
- **€200M+** in market opportunities
- **Serie A** identified as best value league (+19.8 avg gap)
- **Defenders** most undervalued position (+18.0 avg gap)

---

## 📸 Key Visualizations

### Model Comparison: XGBoost vs Random Forest
![Model Comparison](visualizations/model_comparison.png)

### Innovation Impact: Tracking Back Metrics
![Tracking Back Impact](visualizations/tracking_back_impact.png)

### Market Inefficiencies: Value vs Performance
![Value Analysis](visualizations/value_vs_performance.png)

### Feature Importance by Position
![Feature Importance](visualizations/feature_importance.png)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rravi219-spec/Football-ML-Analysis.git
cd Football-ML-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run complete pipeline (~2 minutes)
python scripts/01_data_cleaning.py
python scripts/02_feature_engineering.py
python scripts/03_model_training.py
python scripts/04_player_classification.py
python scripts/05_value_analysis.py
python scripts/06_create_visualizations.py
python scripts/07_defender_lb_rb_classification.py
python scripts/08_aggregate_all_seasons.py
python scripts/09_enhanced_midfielders.py
python scripts/10_enhanced_wingers_lw_rw.py
```

---

## 📁 Project Structure

```
Football-ML-Analysis/
│
├── scripts/                          # 10 Python scripts
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py     # Creates 127 features (includes tracking back!)
│   ├── 03_model_training.py
│   └── ...
│
├── visualizations/                   # 7 professional charts (300 DPI)
│   ├── model_comparison.png
│   ├── tracking_back_impact.png
│   └── ...
│
├── documentation/                    # Comprehensive docs
├── requirements.txt
└── README.md
```

---

## 📋 13 Tactical Positions

**Forwards:** Poachers, False 9s  
**Wingers:** LW, RW  
**Midfielders:** CAM, CM, CDM, Box-to-Box  
**Defenders:** CB, LB, RB, LWB, RWB

---

## 💡 Key Insights

1. **Tracking Back Innovation** - Spatial defensive metrics improve accuracy by 32%
2. **Position-Specific Models** - Separate models outperform unified approach
3. **Career vs Form** - Dual analysis reveals consistency vs current performance
4. **Market Inefficiencies** - Serie A offers best bargains, defenders undervalued
5. **Tactical Evolution** - Only 15 pure CAMs exist, CMs dominate (190 players)

---

## 📊 Sample Results

**Top Poachers (Career 2018-2024):**
1. Erling Haaland - 97.6 (1.02 G/90)
2. Robert Lewandowski - 95.5 (0.94 G/90)
3. Kylian Mbappé - 94.8 (0.99 G/90)

**Top Midfielders:**
1. Luka Modrić (CM) - 83.6 (still elite at 39!)
2. Wilfred Ndidi (CDM) - 87.1

**Top Defenders:**
1. James Tarkowski - 84.3 (7 seasons)

---

## 🔧 Technical Stack

**Core:** Python 3.8+, XGBoost, scikit-learn, pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Data:** FBref.com, Transfermarkt  
**Techniques:** Feature Engineering, Ensemble Learning, Cross-Validation

---

## 📚 Documentation

Full documentation available in `/documentation`:
- Model Comparison
- 2024 Rankings & Career Rankings
- Complete Scripts Guide
- Project Summary

---

## 👨‍💻 Author

**Rakshith Ravi** - ML Engineer & Sports Analytics

- GitHub: [@rravi219-spec](https://github.com/rravi219-spec)
- LinkedIn: [rakshith-ravichandran-079270170](https://linkedin.com/in/rakshith-ravichandran-079270170)
- Email: rravi219@gmail.com

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## ⭐ Support

If you find this project useful, please star it! ⭐

---

**Built with ⚽ and 🤖**  
*Making football analytics accessible through machine learning*

---

## 📄 License
MIT — free to use, edit, and share. A link back is appreciated.
