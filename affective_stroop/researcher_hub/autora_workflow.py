"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from stimulus_sequence import stimulus_sequence

# *** Set up variables *** #
# independent variable is number of training trials (4 - 32)
# dependent variable: emotional_valence_RT_diff (or accuracy_diff)
variables = VariableCollection(
    independent_variables=[Variable(name="n_train", allowed_values=[i for i in range(4, 33)])],
    dependent_variables=[Variable(name="valence_rt_diff", value_range=(-1000, 1000))])

# *** State *** #

# With the variables, we can set up a state. The state object represents the state of our
# closed loop experiment.
state = StandardState(
    variables=variables,
)

# *** Components/Agents *** #
# Components are functions that run on the state. The main components are:
# - theorist
# - experiment-runner
# - experimentalist
# See more about components here: https://autoresearch.github.io/autora/


# ** Theorist ** #
# Here we use a linear regression as theorist, but you can use other theorists included in
# autora (for a list: https://autoresearch.github.io/autora/theorist/)

theorist = LinearRegression()

# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


@on_state()
def theorist_on_state(experiment_data, variables):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]
    return Delta(models=[theorist.fit(x, y)])

# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create a on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(variables, num_samples):
    return Delta(conditions=pool(variables, num_samples))


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key


firebase_credentials = {"type": "service_account",
  "project_id": "stroop-task-cef1d",
  "private_key_id": "2be33a60e31db23a1933e70a1291e95a6a54670a",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCpUgtVcTUEZjPN\n10rnCGhtaDSj26f4HeynyTE1H2ngy87X4kgbbIE3b2CHmLj124U25t4WjmVsgd7x\nEqceA5UKl8vbGL15tEq+K94ihNwYtNsvLvoqAC7izqRhqtA0eMNkNo0/3ah8tOId\nOzrNYRbrKUMcm58jFj+3dORMfdnRJ13dRg9pvPQ4uBK++WfEccGJ1Czsza4piYod\naB5L17DbBIqmhc4Q0nKSy9PZdCORra3OpA3QLJFimVo90Lpr5mRt+Lbc53+caMAk\nrZni63m8x4jaCeIdLlAHdkKBm6XefuWIeAnDDvBjkpJTR04bDr3nyYipM94apOIT\nLY97F7tZAgMBAAECggEABjJc/NMIVDFjWviyopLxs3NZXdfXpWJ++0MHPID6UIJU\n/xymc+58Kim22C+UMVdH4FMBtMSgXKDCJoN+vz/1T+5vRQT6Rj59b8lON2gyX+sa\njHcFK81W6fETHgnw1cLHr965O6milXHPRq+YvtR7UL7xWFYvtEqHqXCrPEts0dU2\nNnqoWGzDQlDC9BlrmZQDFdIrbnyh60f09KDTCwM5g9ZNWJ2og7QhvP25Vw8ElzZ/\nfs0q1GT7ke+veCPP7rmumvkmuidX1IllkGNe6ro4JothSGPFZy9NrUi1rGSrBsnz\nCxPIej5pJl1bcvh9S0DAsGwv9VKxQQ0rNSMkaFjviwKBgQDhKnRZjXBNNf+bMPsJ\njBd+S7vOhdgUJcTx+B20SHxaUZaPstD/n3zceqi1XZpiq86HY4M4S1sJCTKuPQOT\noOXtCO+vQ6RiafxwVFoz/fCaPypVdkC7nzqQPSDPXPfunwlCy/XUAeTYt3ZrFyYT\n5Ng/qpmItkwgiqVUdm5pYl5C6wKBgQDAgdenH16rstQVDfr8FDYdsLMa7hO14aBi\n7hRjhRAI4be2psBELzKH1Q8tjm/2/LQB7+Mg0Ag8vL4Yus6+q6nRJA+E3wFC4v0+\ntL39+qHe0na95qKWnsAoijL6vKUO/MD0OXVlaHsdTL4xxDiuNIAJ4YRHTIuRZveu\nPXMmXVFBywKBgQDcgq1Lfn66vuVCYOeY4/+mChk1GlMQ+CQ2LB07D7no4fonZKHV\nlaW8bsuGQtMNSuCKKuS5XuMaqE1f4hT8oRhL2PKSnBYC43T7tUBZppnZFX8qDxM6\nK6g41gSpz4xnvoxdRE3NgMVTj+iC3DrIRNebEkImZxGK1P0xCIL85f8WXQKBgFxH\n4dDDQOaduvy8zuu35Jkm8zSdm+U8W14RMsTiLGWVjjOIi/Ijjd1/TN9RwGptiPzE\nueQo1UoIUDgalLDiKW2QE8BmBnzBwxQkIE93rdDaowE1Zgs93m+QkA+SDq24i+aH\nBLev9hR9jU7d+S3JDPevm3FySBVTfrePzXs+kI0xAoGBANFVhbUPgMYJy4X9+J3U\n+CsoS0nSZKqgkUhoUqlgei/J7p7DF2T6Et2zSxojl1dyXCcX/FUcyL7qHQVNazXf\nVs0M2msoP6p4hrCprbIkLKkq4goXMrFCWv5OBWJgpN7xIb6K12CNs+YkC/ON7bew\naOO0nTFTsQvQpla83HCpGDtM\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-fbsvc@stroop-task-cef1d.iam.gserviceaccount.com",
  "client_id": "109846230290770747789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40stroop-task-cef1d.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
	
# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=100,
    sleep_time=5)


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    res = []
    for _, condition_row in conditions.iterrows():
        n_train = int(condition_row["n_train"])
        js_code = stimulus_sequence(n_train)
        res.append(js_code)

    conditions_to_send = conditions.copy()
    conditions_to_send["experiment_code"] = res
    data = experiment_runner(conditions_to_send)

    result = [json.loads(item) for item in data]
    return Delta(experiment_data=pd.DataFrame(result))

# Now, we can run our components
for _ in range(3):
    state = experimentalist_on_state(state, num_samples=2)  # Collect 2 conditions per iteration
    state = runner_on_state(state)
    state = theorist_on_state(state)


# *** Report the data *** #
# If you changed the theorist, also change this part
def report_linear_fit(m: LinearRegression, precision=4):
    s = f"y = {np.round(m.coef_[0].item(), precision)} x " \
        f"+ {np.round(m.intercept_.item(), 4)}"
    return s


print(report_linear_fit(state.models[0]))
print(report_linear_fit(state.models[-1]))
