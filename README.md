# DataViz Gallery

A collection of data visualization examples using Python libraries like Matplotlib, Seaborn, and Pandas.

## Enhanced Visualizations

This repository contains both original and enhanced versions of data visualizations for various datasets.

### Enhancements Overview

The enhanced visualizations include the following improvements:

1. **Code Organization and Reusability**
   - Created utility modules with reusable functions
   - Implemented proper documentation with docstrings
   - Added type hints for better code readability
   - Organized code into logical functions with clear purposes

2. **Visual Improvements**
   - Added custom color palettes for better aesthetics
   - Improved layout and spacing
   - Enhanced typography with better fonts and sizes
   - Added informative annotations and data labels
   - Implemented consistent styling across visualizations

3. **Advanced Visualization Techniques**
   - Added kernel density estimates to histograms
   - Created interactive-style visualizations
   - Implemented 3D visualizations for multi-dimensional data
   - Added correlation analysis and heatmaps
   - Created more sophisticated chart types (donut charts, bubble charts)

4. **Data Analysis Enhancements**
   - Added statistical summaries
   - Included correlation analysis
   - Implemented feature importance analysis
   - Added growth rate calculations and comparisons
   - Enhanced data preprocessing and transformation

## Datasets

### Iris Dataset
The classic Iris flower dataset containing measurements for 150 iris flowers from three different species.

- Original notebook: `iris/iris.ipynb`
- Enhanced notebook: `iris/enhanced_iris.ipynb`
- Utility module: `iris/iris_utils.py`

### World Population Dataset
A dataset containing population statistics for countries around the world from 1970 to 2022.

- Original notebook: `world_population/world_analysis.ipynb`
- Enhanced notebook: `world_population/enhanced_world_analysis.ipynb`
- Utility module: `world_population/world_pop_utils.py`

## How to Use

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas matplotlib seaborn numpy
   ```
3. Run the Jupyter notebooks to see the visualizations

## Key Features of Enhanced Visualizations

### Iris Dataset
- Feature distribution histograms with kernel density estimates
- Scatter plots with class means highlighted
- Enhanced pairplot with correlation coefficients
- 3D visualization of feature relationships
- Feature importance analysis
- Correlation heatmaps

### World Population Dataset
- Enhanced bar charts with data labels and formatting
- Donut charts with custom labels and annotations
- Population growth trend lines
- Population density scatter plots with continent-based coloring
- Continent comparison visualizations
- Growth rate analysis

## Dependencies
- Python 3.6+
- pandas
- matplotlib
- seaborn
- numpy