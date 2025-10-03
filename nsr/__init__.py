# triplane, tensorF etc.
from .train_util import TrainLoop3DRec, TrainLoop3DRecTrajVis
from .train_util_cvD import TrainLoop3DcvD

from .cvD.nvsD import TrainLoop3DcvD_nvsD
from .cvD.nvsD_nosr import TrainLoop3DcvD_nvsD_noSR
from .cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD, TrainLoop3DcvD_nvsD_canoD_eg3d
from .cvD.nvsD_canoD_mask import TrainLoop3DcvD_nvsD_canoD_canomask
# from .cvD.nvsD_canoD_nosr import TrainLoop3DcvD_nvsD_canoD
from .cvD.canoD import TrainLoop3DcvD_canoD
from .cvD.nvsD_canoD_multiview import TrainLoop3DcvD_nvsD_canoD_multiview

from .train_util_with_eg3d import TrainLoop3DRecEG3D
from .train_util_with_eg3d_hybrid import TrainLoop3DRecEG3DHybrid
from .train_util_with_eg3d_real import TrainLoop3DRecEG3DReal, TrainLoop3DRecEG3DRealOnly
from .train_util_with_eg3d_hybrid_eg3dD import TrainLoop3DRecEG3DHybridEG3DD
from .train_util_with_eg3d_real_D import TrainLoop3DRecEG3DRealOnl_D

# * difffusion trainer
from .train_util_diffusion import TrainLoop3DDiffusion
from .train_util_diffusion_dit import TrainLoop3DDiffusionDiT
from .train_util_diffusion_single_stage import TrainLoop3DDiffusionSingleStage
# from .train_util_diffusion_single_stage_sds import TrainLoop3DDiffusionSingleStagecvD, TrainLoop3DDiffusionSingleStagecvDSDS
from .train_util_diffusion_single_stage_sds import TrainLoop3DDiffusionSingleStagecvDSDS, TrainLoop3DDiffusionSingleStagecvDSDS_sdswithrec

from .lsgm import *