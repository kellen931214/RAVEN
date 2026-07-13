from diffusers import DPMSolverMultistepScheduler


class LocalDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    """
    Local DPM-Solver scheduler shim used as a no-watermark baseline for
    MAXSIVE_DPM work.

    This intentionally does not add any MAXSIVE frequency template. It lets us
    compare a local scheduler target against the stock DPM target before adding
    watermark-specific behavior.
    """

    pass
