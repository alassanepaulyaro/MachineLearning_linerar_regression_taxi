
import pandas as pd
import tensorflow as tf

import tensorflow_model_analysis as tfma
from google.protobuf import text_format

from tensorflow_model_remediation import min_diff

# Import the dataset
acs_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/acsincome_raw_2018.csv")

# Print five random rows of the pandas DataFrame.
acs_df.sample(5)