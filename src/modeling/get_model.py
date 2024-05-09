from src.modeling.base_ae import BaseAutoEncoder
from src.modeling.ln_gelu_ae import LNAutoEncoder
from src.modeling.xavier_ln_gelu_ae import XAvierLNAutoEncoder
from src.modeling.xavier_gelu_ae import XAvierAutoEncoder
from src.modeling.base_ae_v2 import BaseAutoEncoderV2
from src.modeling.gelu_ae import GELUAutoEncoder
from src.modeling.residual_ae import ResAutoEncoder
from src.modeling.base_ae_no_maxpool import BaseNoMaxPoolAutoEncoder
from src.modeling.base_ae_leaky_relu import LeakyReluBaseAutoEncoder
from src.modeling.base_ae_no_maxpool_v2 import BaseNoMaxPoolAutoEncoderv2

MODEL_DICT = {
    "base_ae": BaseAutoEncoder,
    "ln_ae": LNAutoEncoder,
    "xv_ln_ae": XAvierLNAutoEncoder,
    "xv_ae": XAvierAutoEncoder,
    "base_ae_v2": BaseAutoEncoderV2,
    "gelu_ae": GELUAutoEncoder,
    "residual_ae": ResAutoEncoder, 
    "base_ae_no_maxpool": BaseNoMaxPoolAutoEncoder,
    "base_ae_leaky_relu": LeakyReluBaseAutoEncoder,
    "base_ae_no_maxpool_v2": BaseNoMaxPoolAutoEncoderv2
}


def init_model(model_cfg):
    model_kwargs = (
        {} if model_cfg["model_kwargs"] is None else model_cfg["model_kwargs"]
    )
    model = MODEL_DICT[model_cfg["type"]](model_cfg["model_name"], **model_kwargs)

    return model


def load_model(model_cfg, model_path, device):
    model_kwargs = (
        {} if model_cfg["model_kwargs"] is None else model_cfg["model_kwargs"]
    )

    model = MODEL_DICT[model_cfg["type"]](model_cfg["model_name"], **model_kwargs)
    model.load(device=device, directory=model_path)

    return model
