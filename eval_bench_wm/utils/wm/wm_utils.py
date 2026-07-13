from enum import Enum

from utils.wm.gs_provider import GsProvider
from utils.wm.tr_provider import TrProvider
from utils.wm.prc_provider import PRCProvider
from utils.wm.tag_provider import TagProvider
from utils.wm.ringid_provider import RingIDProvider
from utils.wm.hstr_provider import HSTRProvider
from utils.wm.hsqr_provider import HSQRProvider
from utils.wm.sph_provider import SPHProvider
from utils.wm.t2s_provider import T2SProvider
from utils.wm.maxsive_provider import MaxsiveProvider
from utils.wm.shallow_provider import ShallowProvider
from utils.wm.gm_provider import GmProvider

class WmProviders(Enum):
    GS = GsProvider
    TR = TrProvider
    PRC = PRCProvider
    TAG = TagProvider
    RID = RingIDProvider
    HSTR = HSTRProvider
    HSQR = HSQRProvider
    SPH = SPHProvider
    T2S = T2SProvider
    MAXSIVE = MaxsiveProvider
    SHALLOW = ShallowProvider
    GM = GmProvider

