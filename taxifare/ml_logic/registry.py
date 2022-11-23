from taxifare.ml_logic.params import LOCAL_REGISTRY_PATH
from taxifare.model_target.cloud_model import save_cloud_model
from taxifare.model_target.local_model import save_local_model


import glob
import os
import time
import pickle
import mlflow

from colorama import Fore, Style

from tensorflow.keras import Model, models


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == 'local':
        save_local_model(params, metrics, model, timestamp)

    elif os.environ.get("MODEL_TARGET") == 'gcs':
        save_cloud_model(model, timestamp)

    elif os.environ.get("MODEL_TARGET") == "mlflow":
        print(Fore.BLUE + "\nSave model to mlflow..." + Style.RESET_ALL)

        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        mlflow.set_experiment(experiment_name=mlflow_experiment)

        with mlflow.start_run():

            if params is not None:
                mlflow.log_params(params)

            if metrics is not None:
                mlflow.log_metrics(metrics)

            if model is not None:

                mlflow.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=mlflow_model_name)

        print("\n✅ data saved to mlflow")

        return

    else:
        raise ValueError(f'{os.environ.get("MODEL_TARGET")} not know')

    return None


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    if os.environ.get("MODEL_TARGET") == "mlflow":

        print(Fore.BLUE + "\nLoad model from mlflow..." + Style.RESET_ALL)

        # load model from mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        stage = "Production"

        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")

        try:
            model = mlflow.keras.load_model(model_uri=model_uri)
            print("\n✅ model loaded from mlflow")
        except:
            print(f"\n❌ no model in stage {stage} on mlflow")
            return None

        return model

    if os.environ.get("MODEL_TARGET") == "local":
        # get latest model version
        model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")

        results = glob.glob(f"{model_directory}/*")
        if not results:
            return None

        model_path = sorted(results)[-1]
        print(f"- path: {model_path}")

        model = models.load_model(model_path)
        print("\n✅ model loaded from disk")

        return model
