import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# TODO: Get data.
df = pd.read_csv('...')

# TODO: Parse timestamps, sort timestamps if needed.

# Reset index.
df.reset_index(drop=True, inplace=True)

df.head()

#### Upsample

def upsample_sh(df, timestamp_column, value_column, resolution):
    """
    Upsample values from a data frame with the sample-and-hold technique, and a 
    given resolution.
    
    :param df: the data frame
    :param timestamps_column: the name of the column with the timestamps
    :param values_column: the name of the column with the values
    :param resolution: the time resolution to upsample with
    :type df: pd.DataFrame
    :type timestamps_column: str
    :type values_column: str
    :type resolution: datetime.timedelta
    :returns: upsampled data frame
    :rtype: pd.DataFrame
    """
    
    # Find indices for column names, as we are using itertuples
    # for faster iteration.
    timestamp_index = np.argwhere(df.columns == timestamp_column)[0][0] + 1
    value_index = np.argwhere(df.columns == value_column)[0][0] + 1

    result = []
    previous_row = None

    # Show progress, as this is a potentially long-running process.
    with tqdm(total=len(df), ascii=True, desc='Upsampling') as pbar:
        for row in df.itertuples():
            # Skip first row.
            if previous_row is None:
                previous_row = row
                pbar.update()
                continue

            # Extend range between rows with value.
            timestamps = np.arange(previous_row[timestamp_index], row[timestamp_index], resolution)
            values = np.zeros(len(timestamps)) + previous_row[value_index]

            result.extend(np.array([timestamps, values]).T.tolist())

            # Set previous row, so that we can compare
            # with it.
            previous_row = row
            pbar.update()
            
    # Create data frame.
    return pd.DataFrame(result, columns=[timestamp_column, value_column])

#### Cluster

def cluster1d(df, value_column, label_column, ordered_labels):
    """
    Cluster and label values from a data frame.
    
    :param df: the data frame
    :param value_column: the name of the column with the values
    :param value_column: the name of the column with the labels to be created
    :param ordered_labels: the ordered list of labels to cluster the values with
    :type df: pd.DataFrame
    :type value_column: str
    :type label_column: str
    :type ordered_labels: list
    :returns: labeled data frame
    :rtype: pd.DataFrame
    """
    
    # Create estimator.
    estimator = KMeans(n_clusters=len(ordered_labels))
    
    # Reshapre data and fit estimator.
    estimator.fit(df[value_column].values.reshape(-1, 1))
    
    # Create labels.
    cluster_centers = estimator.cluster_centers_.tolist()
    sorted_cluster_centers = cluster_centers.copy()
    sorted_cluster_centers.sort() 
    
    labels = []
    
    # Show progress, as this is a potentially long-running process.
    with tqdm(total=len(df[value_column]), ascii=True, desc='Clustering') as pbar:
        # Map correct label to the correct cluster centroid. This
        # assumes that cluster centroids are uniquely identifiable.
        for label in estimator.labels_:
            #[sorted_names[sorted_thresholds.index(thresholds[i])] for i in range(len(thresholds))]
            labels.append(ordered_labels[sorted_cluster_centers.index(cluster_centers[label])])
            pbar.update()
            
    # Create data frame and cluster centers.
    df = df.copy()
    df[label_column] = labels
    return df, np.array(sorted_cluster_centers)[:,0]

#### Quantize

def quantize(df, timestamp_column, value_column, label_column, length_column, metric_functions=[], metric_columns=[]):
    """
    Quantize values from a data frame column (pd.Series) and compute additional
    metrics for the sequence.
    
    :param df: the data frame
    :param timestamp_column: the name of the column with the timestamps
    :param value_column: the name of the column with the values
    :param label_column: the name of the column with the labels
    :param length_column: the name of the column with the lengths to be created
    :param metric_functions: the additional and optional metric functions to compute for the sequences
    :param metric_columns: the column names for the additional and optional metric functions
    :type df: pd.DataFrame
    :type value_column: str
    :type label_column: str
    :type label_column: str
    :type metric_functions: list
    :type metric_columns: list
    :returns: labeled data frame
    :rtype: pd.DataFrame
    """
    
    # Check metric functions and columns.
    if len(metric_columns) > 0: 
        if len(metric_functions) != len(metric_columns):
            raise Exception('Length of functions and columns does not match.')
    else:
        # Add default metric columns.
        metric_columns.extend(['Metric{}'.format(i + 1) for i in range(len(metric_functions))])
    
    # Find indices for column names, as we are using itertuples
    # for faster iteration.
    timestamp_index = np.argwhere(df.columns == timestamp_column)[0][0] + 1
    value_index = np.argwhere(df.columns == value_column)[0][0] + 1
    label_index = np.argwhere(df.columns == label_column)[0][0] + 1

    result = []
    previous_rows = []

    # Show progress, as this is a potentially long-running process.
    with tqdm(total=len(df), ascii=True, desc='Quantizing') as pbar:
        for row in df.itertuples():
            # Skip first row of a new run.
            if len(previous_rows) == 0:
                previous_rows = [row]
                pbar.update()
                continue

            # Check if run continues.
            if previous_rows[-1][label_index] != row[label_index]:
                # Get values and convert to NumPy array.
                values = []
                for previous_row in previous_rows:
                    values.append(previous_row[value_index])
                values = np.array(values)

                # Create new row (timestamp, length, label, metrics).
                r = [previous_rows[0][timestamp_index], 
                     row[timestamp_index] - previous_rows[0][timestamp_index], 
                     previous_rows[0][label_index]]
                r.extend([f(values) for f in metric_functions])
                result.append(r)

                # Add current (unused) row to previous rows.
                previous_rows = [row]
            else:
                # Add current row to previous rows.
                previous_rows.append(row)

            pbar.update()
            
    # Create data frame.
    columns = [timestamp_column, length_column, label_column]
    columns.extend(metric_columns)
    return pd.DataFrame(result, columns=columns)

#### Visualize

def step_plot_labels(df, timestamp_column, length_column, label_column, labels, cluster_centers, *args, **kwargs):
    """
    Plot labels in a data frame as a step plot.
    
    :param df: the data frame
    :param timestamp_column: the name of the column with the timestamps
    :param length_column: the name of the column with the lengths
    :param label_column: the name of the column with the labels
    :param labels: the labels
    :param cluster_centers: the cluster centers
    :param args: variable length argument list for Matplotlib
    :param kwargs: arbitrary keyword arguments for Matplotlib
    :type df: pd.DataFrame
    :type timestamp_column: str
    :type length_column: str
    :type label_column: str
    :type labels: list
    :type cluster_centers: list
    """
    xv = []
    yv = []

    for index, row in df.iterrows():
        y = cluster_centers[labels.index(row[label_column])]

        # Create line between start and end of sequence.
        xv.append(row[timestamp_column])
        xv.append(row[timestamp_column] + row[length_column])
        yv.append(y)
        yv.append(y)

    # Plot with additional arguments.
    plt.plot(xv, yv, *args, **kwargs)
    
def step_plot_values(timestamps, values, *args, **kwargs):
    """
    Plot values as a step plot.
    
    :param timestamps: the timestamps
    :param values: the values
    :param args: variable length argument list for Matplotlib
    :param kwargs: arbitrary keyword arguments for Matplotlib
    :type timestamps: list
    :type values: list
    """
        
    # Check length of lists.
    if len(timestamps) != len(values):
        raise Exception('Length of timestamps and values does not match.')
    
    xv = []
    yv = []

    for index in range(len(timestamps)):
        # Skip last timestamp.
        if index == len(timestamps) - 1:
            continue
            
        # Create line between first and second row.
        xv.append(timestamps[index])
        xv.append(timestamps[index + 1])
        yv.append(values[index])
        yv.append(values[index])

    # Plot with additional arguments.
    plt.plot(xv, yv, *args, **kwargs)

def plot(upsampled_df, quantized_df, labels, cluster_centers, start, end):
    """
    Plot upsampled data, quantized data and quantization error for a 
    given time window.
    
    :param upsampled_df: the upsampled data frame
    :param quantized_df: the quantized data frame
    :param labels: the labels
    :param cluster_centers: the cluster centers
    :param start: the start of the time window
    :param end: the end of the time window
    :type upsampled_df: pd.DataFrame
    :type quantized_df: pd.DataFrame
    :type labels: list
    :type cluster_centers: list
    :type start: datetime
    :type end: datetime
    """
    
    # Get sequences.
    quantized_sub_df = quantized_df[(quantized_df['Timestamp'] + quantized_df['Length'] > start) & (quantized_df['Timestamp'] < end)]
    
    # Check data frame.
    if len(quantized_sub_df) == 0:
        raise Exception('Found no sequences in given time window.')

    # Extend start and end, as sequences may start earlier and end later.
    extended_start = quantized_sub_df.iloc[0]['Timestamp']
    extended_end = quantized_sub_df.iloc[-1]['Timestamp'] + quantized_sub_df.iloc[-1]['Length']

    # Get upsampled data.
    upsampled_sub_df = upsampled_df[(upsampled_df['Timestamp'] >= extended_start) & (upsampled_df['Timestamp'] <= extended_end)]
    
    # Plot upsampled data.
    plt.figure(figsize=(20,4))
    step_plot_values(upsampled_sub_df['Timestamp'].values, upsampled_sub_df['Measurement'].values)
    
    # Plot quantized data.
    step_plot_labels(quantized_sub_df, 'Timestamp', 'Length', 'Label', labels, cluster_centers, c='r', linestyle='--')

    plt.legend(['Original', 'Quantized'])
    plt.grid()
    plt.title('Signal ({} to {})'.format(extended_start, extended_end))

    # Plot quantized error.
    plt.figure(figsize=(20,2))
    step_plot_values(upsampled_sub_df['Timestamp'].values, quantize_error(upsampled_sub_df, 'Measurement', cluster_centers) , c='orange')
    plt.grid()
    plt.title('Quantization error')

#### Example

# Create labels.
labels = ['Low', 'High']

# Process data.
upsampled_df = upsample_sh(df, 'Timestamp', 'Measurement', timedelta(seconds=1))
clustered_df, cluster_centers = cluster1d(upsampled_df, 'Measurement', 'Label', labels)
quantized_df = quantize(clustered_df, 'Timestamp', 'Measurement', 'Label', 'Length', [np.min, np.max, np.mean, np.std], ['Min', 'Max', 'Mean', 'SD'])

start = datetime(2018, 4, 17, 8, 39, 50)
end = datetime(2018, 4, 17, 8, 55, 50)
plot(upsampled_df, quantized_df, labels, cluster_centers, start, end)

