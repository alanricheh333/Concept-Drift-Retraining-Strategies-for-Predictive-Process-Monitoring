
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bar_chart(dataset: pd.DataFrame):
    """
    Plot the results in bar chart
    """
    # sampling_option = dataset['Design Option']
    # accuracy = dataset["Accuracy"]
    # f1_score = dataset["F1-Score"]
    
    #labels
    design_option = "Design Option"
    accuracy = "Accuracy"
    sampling_method = "Sampling Method"
    f1_score = "F1-Score"

    baseline_accuracy = dataset.loc[dataset[sampling_method] == "All Eventlog", accuracy].values[0]
    baseline_f1_score = dataset.loc[dataset[sampling_method] == "All Eventlog", f1_score].values[0]


    # Create a bar plot with hue
    # sns.barplot(x='Design Option', y='Accuracy', hue='Sampling Method', data=dataset, palette='pastel')
    # plt.title('Accuracy by Sampling Option')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Sampling Option')
    # plt.xticks(rotation=90, ha='right')
    # plt.legend(title='Sampling Method')
    # plt.show()

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Bar plot for Accuracy
    sns.barplot(x= sampling_method, y= accuracy, hue= design_option, data=dataset, palette='pastel', ax=ax1)
    ax1.set_title(accuracy + ' by ' + design_option)
    ax1.set_ylabel(accuracy)
    ax1.set_xlabel(sampling_method)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    
    ax1_legend = ax1.legend(title=design_option, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax1_legend.set_title(design_option, prop={'size': 7})  # Set title font size
    #ax1.legend(title='Sampling Option', fontsize=3)

    # Add a horizontal line to compare
    ax1.axhline(y=baseline_accuracy, color='black', linestyle='--', linewidth=1)

    # Bar plot for F1 Score
    sns.barplot(x= sampling_method, y= f1_score, hue= design_option, data=dataset, palette='pastel', ax=ax2)
    ax2.set_title(f1_score + ' by ' + design_option)
    ax2.set_ylabel(f1_score)
    ax2.set_xlabel(sampling_method)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right')
    
    ax2_legend = ax2.legend(title=design_option, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax2_legend.set_title(design_option, prop={'size': 7})  # Set title font size
    #ax2.legend(title='Sampling Option', fontsize=3)

    # Add a horizontal line to compare
    ax2.axhline(y=baseline_f1_score, color='black', linestyle='--', linewidth=1) 

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout() # Add tight layout to avoid overlapping labels

    # Place legend outside the plot
    # ax1.legend(title='Sampling Option', bbox_to_anchor=(0, ), loc='upper left')
    # ax2.legend(title='Sampling Option', bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.show()


    # # Set the width of each bar
    # bar_width = 0.35

    # # Calculate the positions of the bars on the x-axis
    # ind = np.arange(len(sampling_option))

    # # Create a bar chart with side-by-side bars
    # plt.figure(figsize=(10, 6))
    # plt.bar(ind, accuracy, width=bar_width, color='blue', label='Accuracy')
    # plt.bar(ind + bar_width, f1_score, width=bar_width, color='green', label='F1-Score')
    # plt.xlabel('Sampling Option')
    # plt.ylabel('Score')
    # plt.title('Accuracy and F1-Score by Sampling Option')
    # plt.xticks(ind + bar_width/2, sampling_option, rotation=90)
    # plt.legend()
    # plt.ylim(0, 1)
    # plt.tight_layout() # Add tight layout to avoid overlapping labels
    # plt.show()


def plot_line_chart(dataset: pd.DataFrame):
    """
    Plot the results in line chart
    """
    # sampling_option = dataset['Design Option']
    # accuracy = dataset["Accuracy"]
    # f1_score = dataset["F1-Score"]

    # # Create a line chart
    # plt.figure(figsize=(10, 6))
    # plt.plot(sampling_option, accuracy, marker='o', color='blue', label='Accuracy')
    # plt.plot(sampling_option, f1_score, marker='o', color='green', label='F1-Score')
    # plt.xlabel('Sampling Option')
    # plt.ylabel('Score')
    # plt.title('Accuracy and F1-Score by Sampling Option')
    # plt.legend()
    # plt.ylim(0, 1)
    # plt.show()

    #labels
    design_option = "Design Option"
    accuracy = "Accuracy"
    sampling_method = "Sampling Method"
    f1_score = "F1-Score"

    baseline_accuracy = dataset.loc[dataset[sampling_method] == "All Eventlog", accuracy].values[0]
    baseline_f1_score = dataset.loc[dataset[sampling_method] == "All Eventlog", f1_score].values[0]

    # Set up the figure
    fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Line plot for Accuracy
    sns.lineplot(x=sampling_method, y=accuracy, hue=design_option, data=dataset, palette='husl', style=design_option, markers=True, ax=ax)
    ax.set_title( accuracy + ' by ' + design_option + ' and ' + sampling_method)
    ax.set_ylabel(accuracy)
    ax.set_xlabel(sampling_method)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    ax.set_xticks(range(len(dataset[sampling_method].unique())))
    ax.set_xticklabels(dataset[sampling_method].unique(), rotation=90, ha='right')

    # Line plot for F1 Score
    sns.lineplot(x=sampling_method, y=f1_score, hue=design_option, data=dataset, palette='husl', style=design_option, markers=True, ax=ax2)
    ax2.set_title(f1_score + ' by ' + design_option + ' and ' + sampling_method)
    ax2.set_ylabel(f1_score)
    ax2.set_xlabel(sampling_method)
    ax2.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    # Add a horizontal line to compare
    ax.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=baseline_f1_score, color='red', linestyle='--', linewidth=1)

    # Add tight layout
    plt.tight_layout()

    plt.show()





