
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.constants import TIMESTAMP_IDENTIRIFIER_CSV


def plot_bar_chart(dataset: pd.DataFrame):
    """
    Plot the results in bar chart
    """
    #labels
    design_option = "Design Option"
    accuracy = "Accuracy"
    sampling_method = "Sampling Method"
    f1_score = "F1-Score"

    baseline_accuracy = dataset.loc[dataset[sampling_method] == "All Eventlog", accuracy].values[0]
    baseline_f1_score = dataset.loc[dataset[sampling_method] == "All Eventlog", f1_score].values[0]

    exp_decay_accuracy = dataset.loc[dataset[sampling_method] == "Exp Decay", accuracy].values[0]
    exp_decay_f1_score = dataset.loc[dataset[sampling_method] == "Exp Decay", f1_score].values[0]

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
    # Add another one
    ax1.axhline(y=exp_decay_accuracy, color='red', linestyle='--', linewidth=1)

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
    # Add another one
    ax2.axhline(y=exp_decay_f1_score, color='red', linestyle='--', linewidth=1)

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout() # Add tight layout to avoid overlapping labels
    
    plt.show()


def plot_line_chart(dataset: pd.DataFrame):
    """
    Plot the results in line chart
    """
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



def plot_drifts(event_log: str):
    """
    Plots event log time line with points in time as drift points.
    Param - event_log: event log name.
    """
    from subprocess import check_output
    import os
    from config import root_directory

    window_size = "100"

    output = check_output(['java', '-jar', os.path.join(root_directory, "core", "concept_drift", "ProDrift2.5.jar"), '-fp', 
                     os.path.join(root_directory, "data", "input", "xes", event_log + ".xes"),
                     '-ddm','events', '-ws', window_size, '-ddnft', '0.0', '-dds', 'high', '-cm', 'activity', '-dcnft', '0.0'], text=True)
    
    result_text = output
    print(result_text)

    #retrieve the useful information from the resulting text
    first_split:list = result_text.split('\n\n\n')
    second_split:list = first_split[0].split('\n\n')
    drift_text_list = []
    drift_text_list.append(second_split[1])
    first_split.pop(0)
    first_split.remove('')
    drift_text_list.extend(first_split)

    #restructre the retrieved information into dict with the event id as key and the event timestamp as value
    drifts_dict = {}
    for item in drift_text_list:
        important_text:str = item.partition("Drift detected at event: ")[2]
        data_info = important_text.split(" ", 1)
        event_num = int(data_info[0])
        event_date = data_info[1][ data_info[1].find("(")+1 : data_info[1].find(")") ]
        drifts_dict[event_num] = event_date
    
    #read the event log in csv format
    dataset = pd.read_csv(os.path.join(root_directory, "data", "input", "csv", event_log + ".csv"))
    
    #get points in time of drifts
    drift_points = []
    for key, value in drifts_dict.items():
        time = dataset.iloc[[key]][TIMESTAMP_IDENTIRIFIER_CSV].values[0]
        drift_points.append(time)

    # Convert timestamp column to datetime object
    dataset[TIMESTAMP_IDENTIRIFIER_CSV] = pd.to_datetime(dataset[TIMESTAMP_IDENTIRIFIER_CSV], format="%Y-%m-%d %H:%M:%S")

    start_time = dataset[TIMESTAMP_IDENTIRIFIER_CSV].min()
    end_time = dataset[TIMESTAMP_IDENTIRIFIER_CSV].max()

    #drift_points = [datetime.datetime.strptime(ts[:-13], "%Y-%m-%d %H:%M:%S") for ts in drift_points]
    drift_points = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in drift_points]

    # add an offset of 0.05 days to the first timestamp
    from datetime import timedelta
    for i in range(4):
        drift_points[i] += timedelta(days=4)


    # set up the figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot([start_time, end_time], [0, 0], color='k', linewidth=2)

    ax.plot(drift_points, np.zeros_like(drift_points), 'o', markersize=8, color='#00994D')

    # add labels
    labels = [str(t)[:10] for t in [start_time, end_time] + list(drift_points)]
    for i, t in enumerate([start_time] + list(drift_points) + [end_time]):
        label = labels[i]
        if i == 0 or i == len(labels) - 1:
            if i % 2 == 0:
                ax.annotate(label, xy=(t, 0), xytext=(0, -50), textcoords='offset points',
                        ha='center', va='center', rotation=90, fontsize=10)
            else:
                ax.annotate(label, xy=(t, 0), xytext=(0, 50), textcoords='offset points',
                        ha='center', va='center', rotation=90, fontsize=10)
        else:
            if i % 2 == 0:
                ax.annotate(label, xy=(t, -0.0025), xytext=(0, -70), textcoords='offset points',
                            ha='center', va='center', rotation=90, fontsize=10,
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=5", color='black'))
            else:
                ax.annotate(label, xy=(t, 0.0025), xytext=(0, 70), textcoords='offset points',
                            ha='center', va='center', rotation=90, fontsize=10,
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=-90,rad=5", color='black'))
        
    #JUST START AND END TIME
    # # add labels
    # labels = [str(t)[:10] for t in [start_time, end_time]]# + list(drift_points)]
    # for i, t in enumerate([start_time, end_time]):# + list(drift_points)):
    #     label = labels[i]
    #     ax.annotate(label, xy=(t, 0), xytext=(0, -50), textcoords='offset points',
    #             ha='center', va='center', rotation=90, fontsize=10)


    # hide y axis and set x axis limits
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(start_time, end_time)

    # ax.vlines([start_time, end_time], -1, 1, colors='r')
    ax.plot([start_time, start_time], [-0.01, 0.01], color='r', linewidth=2)
    ax.plot([end_time, end_time], [-0.01, 0.01], color='r', linewidth=2)


    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim(-0.2, 0.2)

    # remove unnecessary whitespace
    plt.tight_layout()

    # show the plot
    plt.show()
