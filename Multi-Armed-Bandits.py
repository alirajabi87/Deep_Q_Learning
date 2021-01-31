import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


df = pd.DataFrame(np.random.choice(np.arange(0, 2), p=[0.9, 0.1], size=(10000, 5)),
                  columns=list('ABCDE'))

total_views = 10000
