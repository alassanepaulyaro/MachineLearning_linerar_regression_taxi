
# @title Setup - Import relevant modules

import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# @title Import the dataset

training_df = pd.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Get statistics on the dataset.

training_df.describe()

# @title Task 1: Solution (run this code block to view) { display-mode: "form" }

print("""The following columns might contain outliers:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income

In all of those columns:

  * the standard deviation is almost as high as the mean
  * the delta between 75% and max is much higher than the
      delta between min and 25%.""")