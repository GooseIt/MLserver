from ml_server import app
import pickle
import pandas as pd
import numpy as np
import sklearn
from ensembles import MyRandomForest, MyGradBoost


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
