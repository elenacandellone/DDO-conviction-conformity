import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
#from sentence_transformers import SentenceTransformer, util
#from transformers import pipeline
#import torch
import re
from time import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from cmdstanpy import CmdStanModel
from itertools import product
import pickle

#gpt imports
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal

#ideal points imports
import arviz as az
from cmdstanpy import CmdStanModel


np.random.seed(42)

# change the default options of visualization
text_color = "#404040"
custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False, "axes.spines.bottom": False,
                "lines.linewidth": 2, "grid.color": "lightgray", "legend.frameon": False,
                 "xtick.labelcolor": text_color, "ytick.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color,"text.color": text_color,
                "axes.labelcolor": text_color, "axes.titlecolor":text_color,"figure.figsize": [5,3],
                "axes.titlelocation":"left","xaxis.labellocation":"left","yaxis.labellocation":"bottom"}

palette = ["#3d348b","#e6af2e","#191716","#e0e2db"] #use your favourite colours
sns.set_theme(context='paper', style='white', palette=palette, font='Verdana', font_scale=1.1, color_codes=True,
rc=custom_params)
           