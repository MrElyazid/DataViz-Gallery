"""
Utility functions for World Population dataset visualization.
This module provides reusable functions for creating enhanced visualizations
of the World Population dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Optional, Union

# Set default style parameters
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def load_world_population_data(filepath: str) -> pd.DataFrame:
    """
    Load the World Population dataset from a CSV file.
    
    Args:
        filepath: Path to the World Population dataset CSV file
        
    Returns:
        DataFrame containing the World Population dataset
    """
    return pd.read_csv(filepath)

def plot_top_countries_bar(df: pd.DataFrame, n: int = 10, year: str = '2022 Population', 
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create an enhanced bar chart showing the top N countries by population.
    
    Args:
        df: DataFrame containing the World Population dataset
        n: Number of top countries to display
        year: Column name for the population year to use
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Get top N countries
    top_n = df.nlargest(n, year)
    
    # Create a custom color palette with a gradient
    colors = sns.color_palette("viridis", n)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar chart with enhanced styling
    bars = ax.bar(
        top_n['Country/Territory'], 
        top_n[year],
        color=colors,
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        formatted_height = f"{height/1e6:.1f}M" if height < 1e9 else f"{height/1e9:.2f}B"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height * 1.01,
            formatted_height,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            rotation=0
        )
    
    # Add title and labels with enhanced styling
    ax.set_title(f'Top {n} Countries by Population ({year.split()[0]})', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Country', fontsize=14, labelpad=10)
    ax.set_ylabel('Population', fontsize=14, labelpad=10)
    
    # Format y-axis with billions/millions
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Add total population as an annotation
    total_pop = df[year].sum()
    top_n_pop = top_n[year].sum()
    percentage = (top_n_pop / total_pop) * 100
    
    ax.annotate(
        f'These {n} countries represent {percentage:.1f}% of world population',
        xy=(0.5, 0.97),
        xycoords='axes fraction',
        ha='center',
        va='top',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    )
    
    # Add source information
    ax.annotate(
        'Source: World Population Dataset',
        xy=(1.0, -0.12),
        xycoords='axes fraction',
        ha='right',
        va='center',
        fontsize=10,
        fontstyle='italic'
    )
    
    plt.tight_layout()
    
    return fig

def plot_population_pie(df: pd.DataFrame, n: int = 10, year: str = '2022 Population',
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create an enhanced pie chart showing the distribution of population among top N countries.
    
    Args:
        df: DataFrame containing the World Population dataset
        n: Number of top countries to display individually
        year: Column name for the population year to use
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Get top N countries
    top_n = df.nlargest(n, year)
    
    # Calculate total population and "Other" category
    top_n_total = top_n[year].sum()
    other_total = df[year].sum() - top_n_total
    
    # Create a DataFrame for the pie chart
    pie_data = pd.concat([
        top_n[['Country/Territory', year]], 
        pd.DataFrame({'Country/Territory': ['Other'], year: [other_total]})
    ])
    
    # Create a custom color palette
    colors = sns.color_palette("viridis", n+1)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the pie chart with enhanced styling
    wedges, texts, autotexts = ax.pie(
        pie_data[year],
        labels=None,  # We'll add custom labels
        autopct=None,  # We'll add custom percentage labels
        colors=colors,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        startangle=90,
        shadow=True,
        radius=1
    )
    
    # Add a white circle at the center to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.3, fc='white', edgecolor='gray', linewidth=1)
    ax.add_patch(centre_circle)
    
    # Add custom labels with lines
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    kw = dict(arrowprops=dict(arrowstyle="-", color="gray", linewidth=1),
             bbox=bbox_props, zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Calculate label position
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        
        # Format the population value
        pop_value = pie_data[year].iloc[i]
        if pop_value >= 1e9:
            pop_str = f"{pop_value/1e9:.2f}B"
        else:
            pop_str = f"{pop_value/1e6:.1f}M"
        
        # Calculate percentage
        percentage = (pop_value / df[year].sum()) * 100
        
        # Create the label text
        country_name = pie_data['Country/Territory'].iloc[i]
        if len(country_name) > 15:  # Truncate long names
            country_name = country_name[:12] + '...'
            
        label_text = f"{country_name}\n{pop_str} ({percentage:.1f}%)"
        
        # Position the label
        offset = 1.3 if i < len(wedges) - 1 else 1.1  # Different offset for "Other"
        ax.annotate(label_text, xy=(x*offset, y*offset), xytext=(1.5*x, 1.5*y),
                   horizontalalignment=horizontalalignment, **kw)
    
    # Add title with enhanced styling
    ax.set_title(f'Distribution of World Population ({year.split()[0]})', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add total world population in the center
    total_pop = df[year].sum()
    if total_pop >= 1e9:
        total_pop_str = f"{total_pop/1e9:.2f}B"
    else:
        total_pop_str = f"{total_pop/1e6:.1f}M"
    
    ax.text(0, 0, f"Total\n{total_pop_str}", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add source information
    ax.annotate(
        'Source: World Population Dataset',
        xy=(1.0, -0.1),
        xycoords='axes fraction',
        ha='right',
        va='center',
        fontsize=10,
        fontstyle='italic'
    )
    
    plt.tight_layout()
    
    return fig

def plot_population_growth(df: pd.DataFrame, n: int = 5, 
                          years: List[str] = ['1970 Population', '1980 Population', 
                                             '1990 Population', '2000 Population', 
                                             '2010 Population', '2015 Population',
                                             '2020 Population', '2022 Population'],
                          figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a line chart showing population growth over time for the top N countries.
    
    Args:
        df: DataFrame containing the World Population dataset
        n: Number of top countries to display
        years: List of column names for the population years to use
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Get top N countries by most recent population
    top_n = df.nlargest(n, years[-1])
    
    # Extract year values from column names
    year_values = [int(year.split()[0]) for year in years]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a custom color palette
    colors = sns.color_palette("viridis", n)
    
    # Plot each country's population growth
    for i, (_, country_data) in enumerate(top_n.iterrows()):
        country_name = country_data['Country/Territory']
        population_values = [country_data[year] for year in years]
        
        # Plot the line with enhanced styling
        ax.plot(year_values, population_values, 
               marker='o', markersize=8, linewidth=3, 
               color=colors[i], label=country_name)
        
        # Add data point annotations for the most recent year
        ax.annotate(
            f"{population_values[-1]/1e9:.2f}B" if population_values[-1] >= 1e9 else f"{population_values[-1]/1e6:.1f}M",
            xy=(year_values[-1], population_values[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add title and labels with enhanced styling
    ax.set_title(f'Population Growth of Top {n} Countries (1970-2022)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=14, labelpad=10)
    ax.set_ylabel('Population', fontsize=14, labelpad=10)
    
    # Format y-axis with billions
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M')
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Customize x-axis ticks
    plt.xticks(year_values, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend with enhanced styling
    legend = ax.legend(
        title='Countries',
        title_fontsize=12,
        fontsize=11,
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    # Add source information
    ax.annotate(
        'Source: World Population Dataset',
        xy=(1.0, -0.12),
        xycoords='axes fraction',
        ha='right',
        va='center',
        fontsize=10,
        fontstyle='italic'
    )
    
    plt.tight_layout()
    
    return fig

def plot_population_density_scatter(df: pd.DataFrame, figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a scatter plot showing the relationship between population and density,
    with point size representing land area.
    
    Args:
        df: DataFrame containing the World Population dataset
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out very small countries for better visualization
    filtered_df = df[df['Area (km²)'] > 1000].copy()
    
    # Create a normalized size array for the scatter points based on area
    area_sizes = np.sqrt(filtered_df['Area (km²)']) / 5
    area_sizes = np.clip(area_sizes, 10, 500)  # Limit the size range
    
    # Create a colormap based on continent
    continents = filtered_df['Continent'].unique()
    continent_colors = dict(zip(continents, sns.color_palette("viridis", len(continents))))
    point_colors = [continent_colors[cont] for cont in filtered_df['Continent']]
    
    # Create the scatter plot with enhanced styling
    scatter = ax.scatter(
        filtered_df['2022 Population'], 
        filtered_df['Density (per km²)'],
        s=area_sizes,
        c=point_colors,
        alpha=0.7,
        edgecolors='white',
        linewidth=1
    )
    
    # Add labels for the largest countries or those with extreme values
    top_by_pop = filtered_df.nlargest(5, '2022 Population')
    top_by_density = filtered_df.nlargest(5, 'Density (per km²)')
    top_by_area = filtered_df.nlargest(5, 'Area (km²)')
    
    # Combine and remove duplicates
    countries_to_label = pd.concat([top_by_pop, top_by_density, top_by_area]).drop_duplicates()
    
    for _, country in countries_to_label.iterrows():
        ax.annotate(
            country['Country/Territory'],
            xy=(country['2022 Population'], country['Density (per km²)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add title and labels with enhanced styling
    ax.set_title('Population vs Population Density (2022)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Population', fontsize=14, labelpad=10)
    ax.set_ylabel('Population Density (per km²)', fontsize=14, labelpad=10)
    
    # Use logarithmic scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Format x-axis with billions/millions
    ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.1f}M')
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Create a custom legend for continents
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 label=continent)
                      for continent, color in continent_colors.items()]
    
    # Add a second legend for area size reference
    area_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, label='10,000 km²'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=12, label='100,000 km²'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=16, label='1,000,000 km²')
    ]
    
    # Add both legends
    continent_legend = ax.legend(
        handles=legend_elements,
        title='Continent',
        title_fontsize=12,
        fontsize=10,
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    ax.add_artist(continent_legend)
    
    ax.legend(
        handles=area_legend_elements,
        title='Land Area',
        title_fontsize=12,
        fontsize=10,
        loc='lower left',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    # Add source information
    ax.annotate(
        'Source: World Population Dataset',
        xy=(1.0, -0.12),
        xycoords='axes fraction',
        ha='right',
        va='center',
        fontsize=10,
        fontstyle='italic'
    )
    
    plt.tight_layout()
    
    return fig

def plot_continent_population_comparison(df: pd.DataFrame, 
                                        years: List[str] = ['1970 Population', '2022 Population'],
                                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a grouped bar chart comparing population by continent for different years.
    
    Args:
        df: DataFrame containing the World Population dataset
        years: List of column names for the population years to compare
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Matplotlib Figure object
    """
    # Group by continent and sum the population for each year
    continent_data = df.groupby('Continent')[years].sum().reset_index()
    
    # Sort by the most recent year's population
    continent_data = continent_data.sort_values(by=years[-1], ascending=False)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the width of the bars and positions
    bar_width = 0.8 / len(years)
    positions = np.arange(len(continent_data))
    
    # Create a custom color palette
    colors = sns.color_palette("viridis", len(years))
    
    # Create the grouped bar chart
    for i, year in enumerate(years):
        # Calculate the position for this group of bars
        pos = positions + (i - len(years)/2 + 0.5) * bar_width
        
        # Create the bars
        bars = ax.bar(
            pos,
            continent_data[year],
            width=bar_width,
            color=colors[i],
            edgecolor='black',
            linewidth=1,
            alpha=0.8,
            label=year.split()[0]
        )
        
        # Add data labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            formatted_height = f"{height/1e9:.2f}B" if height >= 1e9 else f"{height/1e6:.0f}M"
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height * 1.01,
                formatted_height,
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                rotation=0
            )
    
    # Add title and labels with enhanced styling
    year_range = f"{years[0].split()[0]}-{years[-1].split()[0]}"
    ax.set_title(f'Population by Continent: {year_range}', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Continent', fontsize=14, labelpad=10)
    ax.set_ylabel('Population', fontsize=14, labelpad=10)
    
    # Set x-axis ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(continent_data['Continent'], fontsize=12)
    
    # Format y-axis with billions
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M')
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Add legend with enhanced styling
    legend = ax.legend(
        title='Year',
        title_fontsize=12,
        fontsize=11,
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    # Calculate and display growth percentages
    for i, continent in enumerate(continent_data['Continent']):
        start_pop = continent_data[years[0]].iloc[i]
        end_pop = continent_data[years[-1]].iloc[i]
        growth_pct = ((end_pop / start_pop) - 1) * 100
        
        ax.annotate(
            f"+{growth_pct:.1f}%",
            xy=(positions[i], max(start_pop, end_pop) * 1.1),
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add source information
    ax.annotate(
        'Source: World Population Dataset',
        xy=(1.0, -0.12),
        xycoords='axes fraction',
        ha='right',
        va='center',
        fontsize=10,
        fontstyle='italic'
    )
    
    plt.tight_layout()
    
    return fig